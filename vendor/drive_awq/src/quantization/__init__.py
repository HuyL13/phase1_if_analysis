"""Vendored AWQ quantization entry points used by Phase 1."""

from src.quantization.awq import AWQConfig, AWQQuantizerXL
from src.quantization.state import IntegerQuantizedTensorState

__all__ = ["AWQConfig", "AWQQuantizerXL", "IntegerQuantizedTensorState"]
