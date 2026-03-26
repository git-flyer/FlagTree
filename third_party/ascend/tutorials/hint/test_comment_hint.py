# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Comment Hint Test
=================

Tests the #@hint: comment annotation mechanism for the Ascend backend.

This verifies that:
1. #@hint:dot_pad_only_k on tl.load lines generates AnnotationOp with dot_pad_only_k attr in TTIR
2. #@hint:bind_sub_block on for loops generates bind_sub_block attr on scf.for in TTIR
3. The kernel compiles and runs correctly end-to-end
"""

import torch
import torch_npu

import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, ascend
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions


# ---------------------------------------------------------------------------
# Kernel with #@hint:dot_pad_only_k on tl.load
# ---------------------------------------------------------------------------
@triton.jit
def matmul_hint_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):  #@hint:bind_sub_block
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M and k_mask[None, :], other=0.0)  #@hint:dot_pad_only_k
        b = tl.load(b_ptrs, mask=k_mask[:, None] and offs_n[None, :] < N, other=0.0)  #@hint:dot_pad_only_k
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# Helper: compile kernel to TTIR string for IR inspection
# ---------------------------------------------------------------------------
def get_ttir_str(kernel_fn, signature, constants):
    src = ASTSource(kernel_fn, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend.load_dialects(context)
    options = NPUOptions()
    ttir = ast_to_ttir(kernel_fn, src, context, options, {}, {})
    return str(ttir)


# ---------------------------------------------------------------------------
# Test 1: Verify IR contains hint annotations
# ---------------------------------------------------------------------------
def test_ir_hint_annotations():
    print("=" * 60)
    print("Test 1: Verify IR hint annotations")
    print("=" * 60)

    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    constants = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}

    ttir_str = get_ttir_str(matmul_hint_kernel, signature, constants)

    # Check for dot_pad_only_k annotation in IR
    has_dot_pad = "dot_pad_only_k" in ttir_str
    # Check for bind_sub_block attribute on for op in IR
    has_bind_sub = "bind_sub_block" in ttir_str

    print(f"  dot_pad_only_k found in IR: {has_dot_pad}")
    print(f"  bind_sub_block found in IR: {has_bind_sub}")

    if has_dot_pad:
        print("  [PASS] dot_pad_only_k hint correctly attached to IR")
    else:
        print("  [WARN] dot_pad_only_k not found in IR - hint may not have been processed")

    if has_bind_sub:
        print("  [PASS] bind_sub_block hint correctly attached to IR")
    else:
        print("  [WARN] bind_sub_block not found in IR - hint may not have been processed")

    # Print a snippet of the IR for debugging
    print("\n--- TTIR snippet (first 2000 chars) ---")
    print(ttir_str[:2000])
    print("--- end ---\n")

    assert has_dot_pad, "dot_pad_only_k annotation not found in generated TTIR"
    assert has_bind_sub, "bind_sub_block attribute not found in generated TTIR"
    print("  [PASS] All IR hint checks passed\n")


# ---------------------------------------------------------------------------
# Test 2: End-to-end matmul with hints - verify correctness
# ---------------------------------------------------------------------------
def test_e2e_matmul_with_hints():
    print("=" * 60)
    print("Test 2: End-to-end matmul with comment hints")
    print("=" * 60)

    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

    torch.manual_seed(0)
    a = torch.randn((M, K), device='npu', dtype=torch.float16)
    b = torch.randn((K, N), device='npu', dtype=torch.float16)
    c = torch.empty((M, N), device='npu', dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_hint_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    c_ref = torch.matmul(a, b)
    max_diff = torch.max(torch.abs(c.float() - c_ref.float())).item()
    print(f"  Max difference between triton and torch: {max_diff}")

    # fp16 matmul tolerance
    assert max_diff < 1.0, f"Result mismatch: max_diff={max_diff} exceeds tolerance"
    print("  [PASS] End-to-end matmul result is correct\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_ir_hint_annotations()
    test_e2e_matmul_with_hints()
    print("All comment hint tests passed!")
