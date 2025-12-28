# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Backward compatibility shim.

This module has been unified into vllm.utils.flashinfer.
Please update your imports to use:

    from vllm.utils.flashinfer import (
        FlashinferMoeBackend,
        get_flashinfer_moe_backend,
        rotate_flashinfer_fp8_moe_weights,
        # ... etc
    )

This shim will be removed in a future release.
"""

import warnings

# Re-export all symbols from the unified location
from vllm.utils.flashinfer import (
    FlashinferMoeBackend,
    apply_flashinfer_per_tensor_scale_fp8,
    build_flashinfer_fp8_cutlass_moe_prepare_finalize,
    calculate_tile_tokens_dim,
    flashinfer_cutlass_moe_fp8,
    get_flashinfer_moe_backend,
    get_moe_scaling_factors,
    is_flashinfer_supporting_global_sf,
    register_moe_scaling_factors,
    rotate_flashinfer_fp8_moe_weights,
    select_cutlass_fp8_gemm_impl,
    swap_w13_to_w31,
)

warnings.warn(
    "vllm.model_executor.layers.quantization.utils.flashinfer_utils is deprecated. "
    "Please use vllm.utils.flashinfer instead. "
    "This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FlashinferMoeBackend",
    "calculate_tile_tokens_dim",
    "swap_w13_to_w31",
    "rotate_flashinfer_fp8_moe_weights",
    "apply_flashinfer_per_tensor_scale_fp8",
    "get_moe_scaling_factors",
    "register_moe_scaling_factors",
    "build_flashinfer_fp8_cutlass_moe_prepare_finalize",
    "select_cutlass_fp8_gemm_impl",
    "flashinfer_cutlass_moe_fp8",
    "get_flashinfer_moe_backend",
    "is_flashinfer_supporting_global_sf",
]
