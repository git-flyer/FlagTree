# flagtree tle
"""
TLE (Triton Language Extensions) Semantic Analysis Module

This module provides semantic analysis and type checking for TLE operations
"""

from __future__ import annotations
import warnings
from typing import List, Optional, Sequence, Tuple, Union

from triton._C.libtriton import ir
from triton import language as tl
from . import types as tle


class TLESemanticError(Exception):
    """TLE operation semantic error exception class"""

    def __init__(self, message: str, operation: str = None):
        self.operation = operation
        self.message = message
        if operation:
            super().__init__(f"TLE semantic error in {operation}: {message}")
        else:
            super().__init__(f"TLE semantic error: {message}")


class TLESemantic:
    """Semantic analyzer for TLE operations"""

    def __init__(self, builder: ir.builder):
        self.builder = builder

    def validate_alloc_shape(self, shape: Sequence[Union[int, any]]) -> List[int]:
        """Validate allocation shape"""
        if not shape:
            raise TLESemanticError("Allocation shape cannot be empty", "alloc")

        unwrapped_shape = []
        for i, dim in enumerate(shape):
            if hasattr(dim, 'value'):  # constexpr-like objects
                dim_val = dim.value
            elif isinstance(dim, int):
                dim_val = dim
            else:
                raise TLESemanticError(f"Shape dimension {i} must be integer or constexpr-like, but got {type(dim)}",
                                       "alloc")

            if dim_val <= 0:
                raise TLESemanticError(f"Shape dimension {i} must be positive, but got {dim_val}", "alloc")

            unwrapped_shape.append(dim_val)

        return unwrapped_shape

    def validate_alloc_dtype(self, dtype: tl.dtype) -> tl.dtype:
        """Validate allocation data type"""
        if not isinstance(dtype, tl.dtype):
            raise TLESemanticError(f"Data type must be tl.dtype, but got {type(dtype)}", "alloc")

        supported_types = [
            tl.float32, tl.float16, tl.bfloat16, tl.int8, tl.int16, tl.int32, tl.int64, tl.uint8, tl.uint16, tl.uint32,
            tl.uint64, tl.int1  # boolean type equivalent in Triton
        ]

        if dtype not in supported_types:
            warnings.warn(f"Data type {dtype} may not be fully supported, recommend using standard data types",
                          UserWarning)

        return dtype

    def validate_copy_compatibility(self, src: tl.tensor, dst: tle.buffered_tensor,
                                    copy_shape: Sequence[Union[int, tl.constexpr]]) -> None:

        if src.type.element_ty != dst.type.element_ty:
            raise TLESemanticError(
                f"Source data type {src.type.element_ty} incompatible with destination data type {dst.type.element_ty}",
                "copy")

        copy_shape_unwrapped = [dim.value if hasattr(dim, 'value') else dim for dim in copy_shape]
        src_shape = list(src.type.shape)
        dst_shape = list(dst.type.shape)

        for i, copy_dim in enumerate(copy_shape_unwrapped):
            if i < len(src_shape) and copy_dim > src_shape[i]:
                raise TLESemanticError(f"Copy dimension {i} ({copy_dim}) exceeds source tensor range ({src_shape[i]})",
                                       "copy")
            if i < len(dst_shape) and copy_dim > dst_shape[i]:
                raise TLESemanticError(
                    f"Copy dimension {i} ({copy_dim}) exceeds destination buffer range ({dst_shape[i]})", "copy")

    def validate_local_load_buffer(self, buffer: tle.buffered_tensor) -> None:
        """Validate local load buffer"""
        if not isinstance(buffer, tle.buffered_tensor):
            raise TLESemanticError(f"Buffer must be tle.buffered_tensor, but got {type(buffer)}", "local_load")


    def analyze_alloc_operation(self, shape: Sequence[Union[int, any]], dtype: tl.dtype,
                                layout: Optional[tle.shared_layout], storage: tle.scope) -> Tuple[List[int], tl.dtype]:
        """Analyze alloc operation semantics"""
        validated_shape = self.validate_alloc_shape(shape)
        validated_dtype = self.validate_alloc_dtype(dtype)
        return validated_shape, validated_dtype

    def analyze_copy_operation(self, src: tl.tensor, dst: tle.buffered_tensor,
                               copy_shape: Sequence[Union[int, any]]) -> None:
        """Analyze copy operation semantics"""
        self.validate_copy_compatibility(src, dst, copy_shape)

    def analyze_local_load_operation(self, buffer: tle.buffered_tensor) -> None:
        """Analyze local_load operation semantics"""
        self.validate_local_load_buffer(buffer)

    def validate_extract_tile_params(
        self, 
        src: tl.tensor, 
        offsets: Sequence[int], 
        tile_shape: Sequence[int]
    ) -> None:
        """
        Validate extract_tile parameters.
    
        Checks:
            1. src is tl.tensor
            2. offsets and tile_shape are non-empty
            3. All values are integers
            4. Dimensions match
            5. Offsets are non-negative
            6. Tile fits within source bounds
        """
        # ✅ 检查1: src类型
        if not isinstance(src, tl.tensor):
            raise TLESemanticError(
                f"Source must be tl.tensor, but got {type(src)}", 
                "extract_tile"
            )
    
        # ✅ 检查2: 非空
        if not offsets or not tile_shape:
            raise TLESemanticError(
                "Offsets and tile_shape cannot be empty", 
                "extract_tile"
            )
    
        # ✅ 检查3: 解包并验证类型
        offsets_unwrapped = [
            o.value if hasattr(o, 'value') else o 
            for o in offsets
        ]
        tile_shape_unwrapped = [
            s.value if hasattr(s, 'value') else s 
            for s in tile_shape
        ]
    
        if any(not isinstance(o, int) for o in offsets_unwrapped):
            raise TLESemanticError(
                "All offsets must be int or constexpr", 
                "extract_tile"
            )
    
        if any(not isinstance(s, int) for s in tile_shape_unwrapped):
            raise TLESemanticError(
                "All tile_shape dims must be int or constexpr", 
                "extract_tile"
            )
    
        # ✅ 检查4: 正数
        if any(s <= 0 for s in tile_shape_unwrapped):
            raise TLESemanticError(
                "All tile_shape dims must be positive", 
                "extract_tile"
            )
    
        # ✅ 检查5: 非负
        if any(o < 0 for o in offsets_unwrapped):
            raise TLESemanticError(
                "All offsets must be non-negative", 
                "extract_tile"
            )
    
        # ✅ 检查6: 维度匹配
        src_shape = list(src.type.shape)
    
        if len(offsets_unwrapped) != len(src_shape):
            raise TLESemanticError(
                f"Offsets rank ({len(offsets_unwrapped)}) must match "
                f"source rank ({len(src_shape)})", 
                "extract_tile"
            )
    
        if len(tile_shape_unwrapped) != len(src_shape):
            raise TLESemanticError(
                f"Tile_shape rank ({len(tile_shape_unwrapped)}) must match "
                f"source rank ({len(src_shape)})", 
                "extract_tile"
            )
    
        # ✅ 检查7: 边界（如果源shape是静态的）
        if all(isinstance(dim, int) for dim in src_shape):
            for i, (offset, tile_dim, src_dim) in enumerate(
                zip(offsets_unwrapped, tile_shape_unwrapped, src_shape)
            ):
                if offset + tile_dim > src_dim:
                    raise TLESemanticError(
                        f"Dimension {i}: offset({offset}) + tile_shape({tile_dim}) "
                        f"> source({src_dim})", 
                        "extract_tile"
                    )


    def analyze_extract_tile_operation(
        self, 
        src: tl.tensor, 
        offsets: Sequence[int], 
        tile_shape: Sequence[int]
    ) -> None:
        """Analyze extract_tile operation semantics"""
        self.validate_extract_tile_params(src, offsets, tile_shape)
