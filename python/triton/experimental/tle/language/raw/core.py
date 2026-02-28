import triton.language as tl
from triton.language.core import builtin, tensor


@builtin
def call(func, outputs, inputs, _semantic=None):
    results = _semantic.builder.create_tle_raw_call(f"{func.llvm}", [output.handle for output in outputs],
                                                    [input.handle for input in inputs])
    tensors = [tensor(result, output.type) for result, output in zip(results, outputs)]
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tl.tuple(tensors)
