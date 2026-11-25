"""SOLO: sample-faithful and detectable LLM watermarking under repeated sampling (content-anchored single-owner contexts)."""
from .context import SoloConfig, context
from .detect import Detection, calibrated_threshold, detect
from .sampler import SoloLogitsProcessor, SoloSampler, generate

__all__ = ["SoloConfig", "context", "Detection", "detect", "calibrated_threshold", "SoloSampler", "SoloLogitsProcessor",
           "generate"]
__version__ = "0.1.0"
