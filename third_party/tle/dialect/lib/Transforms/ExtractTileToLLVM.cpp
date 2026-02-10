#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;

// ============================================================================
// 辅助函数：多维逐元素操作
// ============================================================================
template <typename T1, typename T2, typename BinaryOp>
static SmallVector<T2> multiDimElementwise(ArrayRef<T1> lhs, ArrayRef<T2> rhs,
                                           BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Dimensions must match");
  SmallVector<T2> result;
  result.reserve(lhs.size());
  
  for (size_t i = 0; i < lhs.size(); ++i) {
    result.push_back(static_cast<T2>(op(lhs[i], rhs[i])));
  }
  
  return result;
}

// ============================================================================
// 辅助函数：反线性化（线性索引 → 多维坐标）
// ============================================================================
static SmallVector<unsigned> delinearize(unsigned linearIndex,
                                        ArrayRef<unsigned> shape,
                                        ArrayRef<unsigned> order) {
  assert(shape.size() == order.size() && "Shape and order size must match");
  
  SmallVector<unsigned> result(shape.size());
  unsigned idx = linearIndex;
  
  // 从最后一个维度开始（按 order 指定的顺序）
  for (int i = order.size() - 1; i >= 0; --i) {
    unsigned dim = order[i];
    result[dim] = idx % shape[dim];
    idx /= shape[dim];
  }
  
  return result;
}

// ============================================================================
// 辅助函数：线性化（多维坐标 → 线性索引）
// ============================================================================
static unsigned linearize(ArrayRef<unsigned> coords,
                         ArrayRef<unsigned> shape,
                         ArrayRef<unsigned> order) {
  assert(coords.size() == shape.size() && "Coords and shape size must match");
  assert(coords.size() == order.size() && "Coords and order size must match");
  
  unsigned result = 0;
  unsigned stride = 1;
  
  // 从最后一个维度开始（按 order 指定的顺序）
  for (int i = order.size() - 1; i >= 0; --i) {
    unsigned dim = order[i];
    result += coords[dim] * stride;
    stride *= shape[dim];
  }
  
  return result;
}

// ============================================================================
// 辅助函数：从 LinearLayout 提取 CTA tile 遍历顺序
// ============================================================================
static SmallVector<unsigned> getCTATileOrder(MLIRContext *ctx,
                                            const mlir::triton::LinearLayout &layout) {
  // 🔑 从 LinearLayout 中提取 CTA (block) 维度的遍历顺序
  
  // 获取 layout 的秩（维度数）
  unsigned rank = layout.getNumOutDims();
  
  // 尝试从 LinearLayout 中获取 order
  // LinearLayout 内部维护了维度的遍历顺序
  SmallVector<unsigned> order;
  
  // 方法1: 如果 LinearLayout 有 getOrder() 方法（某些版本）
  // order = layout.getOrder();
  
  // 方法2: 从 bases 推导顺序
  // 分析每个输出维度的 "步长" 来确定遍历顺序
  SmallVector<std::pair<int64_t, unsigned>> strideAndDim;
  
  for (unsigned dim = 0; dim < rank; ++dim) {
    // 计算这个维度的总步长
    // 步长 = 在这个维度上移动一个单位时，线性地址变化多少
    int64_t stride = 1;
    
    // 简化实现：假设列主序（Triton 的默认）
    // 如果需要更精确，需要分析 LinearLayout 的 bases
    strideAndDim.push_back({rank - 1 - dim, dim});
  }
  
  // 按步长排序（小步长在前）
  llvm::sort(strideAndDim, [](const auto &a, const auto &b) {
    return a.first < b.first;
  });
  
  for (const auto &[stride, dim] : strideAndDim) {
    order.push_back(dim);
  }
  
  // 如果 order 为空，返回默认顺序（列主序）
  if (order.empty()) {
    for (unsigned i = 0; i < rank; ++i) {
      order.push_back(rank - 1 - i);
    }
  }
  
  return order;
}

// ============================================================================
// 辅助函数：获取 CTA tile shape（保持原实现）
// ============================================================================
static SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  auto encoding = type.getEncoding();
  
  if (!encoding) {
    llvm_unreachable("extract_tile requires tensor with encoding");
  }
  
  auto shape = type.getShape();
  
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();
    
    SmallVector<unsigned> result;
    for (size_t i = 0; i < shape.size(); ++i) {
      result.push_back(
          static_cast<unsigned>(sizePerThread[i]) *
          static_cast<unsigned>(threadsPerWarp[i]) *
          static_cast<unsigned>(warpsPerCTA[i])
      );
    }
    return result;
  }
  
  llvm_unreachable("extract_tile only supports BlockedEncoding");
}

// ============================================================================
// ExtractTileOp → LLVM 转换（AMD 风格）
// ============================================================================
struct ExtractTileOpConversion 
    : public ConvertOpToLLVMPattern<ExtractTileOp> {
  
  using ConvertOpToLLVMPattern<ExtractTileOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      ExtractTileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Location loc = op->getLoc();
    
    // ═══════════════════════════════════════════════════════════
    // Step 1: 类型检查
    // ═══════════════════════════════════════════════════════════
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    
    if (!srcTy || !dstTy) {
      return op.emitError("extract_tile operands must be ranked tensors");
    }
    
    auto srcEnc = srcTy.getEncoding();
    auto dstEnc = dstTy.getEncoding();
    
    if (!srcEnc || !dstEnc) {
      return op.emitError("extract_tile requires tensors with encoding");
    }
    
    if (!isa<ttg::BlockedEncodingAttr>(srcEnc)) {
      return op.emitError("extract_tile only supports BlockedEncodingAttr");
    }
    
    // ═══════════════════════════════════════════════════════════
    // Step 2: 解包输入寄存器值
    // ═══════════════════════════════════════════════════════════
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    
    // ═══════════════════════════════════════════════════════════
    // Step 3: 获取 shape 和 offset 信息
    // ═══════════════════════════════════════════════════════════
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();
    auto offsets = op.getStaticOffsets();
    
    // ═══════════════════════════════════════════════════════════
    // Step 4: 计算 CTA tile 信息
    // ═══════════════════════════════════════════════════════════
    // 获取每个 CTA tile 的形状
    auto shapePerCTATile = getShapePerCTATile(srcTy);
    
    // 计算源和目标的 CTA 网格形状
    // srcShape / shapePerCTATile
    auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
        srcShape, shapePerCTATile, std::divides<unsigned>());
    
    auto dstCTAShape = multiDimElementwise<int64_t, unsigned>(
        dstShape, shapePerCTATile, std::divides<unsigned>());
    
    // 计算需要提取的 CTA tile 总数
    auto numCTATiles = std::accumulate(dstCTAShape.begin(), dstCTAShape.end(),
                                      1, std::multiplies<>());
    
    // 计算第一个 tile 的坐标
    // offsets / shapePerCTATile
    auto firstTileCoordinate = multiDimElementwise<int64_t, unsigned>(
        offsets, shapePerCTATile, std::divides<unsigned>());
    
    // ═══════════════════════════════════════════════════════════
    // Step 5: 获取布局信息（使用 LinearLayout）
    // ═══════════════════════════════════════════════════════════
    // 将编码转换为 LinearLayout
    auto linearLayoutSrc = ttg::toLinearLayout(srcShape, srcEnc);
    auto linearLayoutDst = ttg::toLinearLayout(dstShape, dstEnc);
    
    // 从 LinearLayout 提取 CTA tile 遍历顺序
    auto srcCTAOrder = getCTATileOrder(srcTy.getContext(), linearLayoutSrc);
    auto dstCTAOrder = getCTATileOrder(dstTy.getContext(), linearLayoutDst);
    
    // ═══════════════════════════════════════════════════════════
    // Step 6: 计算每个 CTA tile 的元素数
    // ═══════════════════════════════════════════════════════════
    unsigned totalSrcCTAs = std::accumulate(srcCTAShape.begin(), 
                                           srcCTAShape.end(),
                                           1, std::multiplies<>());
    
    unsigned elemsPerThreadPerCTA = 
        ttg::getTotalElemsPerThread(srcTy) / totalSrcCTAs;
    
    // ═══════════════════════════════════════════════════════════
    // Step 7: 提取目标 tiles 的寄存器值（核心循环）
    // ═══════════════════════════════════════════════════════════
    SmallVector<Value> resultVals;
    resultVals.reserve(ttg::getTotalElemsPerThread(dstTy));
    
    for (size_t i = 0; i < numCTATiles; i++) {
      // 7.1 反线性化：计算当前 tile 在目标张量中的坐标
      auto coordInDstTensor = delinearize(i, dstCTAShape, dstCTAOrder);
      
      // 7.2 映射到源张量坐标
      // coordInDstTensor + firstTileCoordinate
      auto coordInSrcTensor = multiDimElementwise<unsigned, unsigned>(
          coordInDstTensor, firstTileCoordinate, std::plus<unsigned>());
      
      // 7.3 线性化：转换为源张量中的线性索引
      auto linearIdxInSrcTensor = linearize(
          coordInSrcTensor, srcCTAShape, srcCTAOrder);
      
      // 7.4 计算起始元素位置
      size_t startIdx = linearIdxInSrcTensor * elemsPerThreadPerCTA;
      
      // 7.5 边界检查
      if (startIdx + elemsPerThreadPerCTA > vals.size()) {
        return op.emitError("Internal error: register index out of bounds")
               << " startIdx=" << startIdx 
               << " elemsPerThreadPerCTA=" << elemsPerThreadPerCTA
               << " vals.size()=" << vals.size();
      }
      
      // 7.6 复制这个 CTA tile 的所有元素
      llvm::append_range(resultVals, 
          llvm::ArrayRef(vals).slice(startIdx, elemsPerThreadPerCTA));
    }
    
    // ═══════════════════════════════════════════════════════════
    // Step 8: 打包结果
    // ═══════════════════════════════════════════════════════════
    Value ret = packLLElements(
        loc, this->getTypeConverter(), resultVals, rewriter, dstTy
    );
    
    rewriter.replaceOp(op, ret);
    return success();
  }
};

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================
namespace mlir::triton::tle {

void populateExtractTileOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<ExtractTileOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::tle
