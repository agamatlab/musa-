"""Opt-in: store nn.Linear weights in BF16 so BF16 autocast stops re-casting them.

Under torch.autocast(dtype=bfloat16), every Linear casts its FP32 weight and bias to
BF16 on every call (~430 cast kernels, ~7 ms per ESM3 forward on the S5000). The cast
is deterministic, so converting the Linear parameters once gives the matmuls exactly
the same BF16 operands: outputs stay bit-identical to FP32-weights + autocast.

Everything autocast keeps in FP32 (LayerNorm, Embedding, scalar parameters, the FP32
residual stream) is left untouched. The model must then always run under BF16 autocast;
a BF16 Linear fed an FP32 input outside autocast raises a dtype error.
"""
import torch
import torch.nn as nn


def prepare_bf16_linears(model: nn.Module) -> int:
    """Convert the weight/bias of every nn.Linear in `model` to BF16 in place.

    Returns the number of Linear modules converted.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.to(torch.bfloat16)
            n += 1
    return n
