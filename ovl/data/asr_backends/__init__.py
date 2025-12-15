"""ASR backend adapters for universal dataset maker"""

from .base import ASRBackend, TranscriptionResult
from .huggingface_backend import HuggingFaceASRBackend

__all__ = [
    "ASRBackend",
    "TranscriptionResult",
    "HuggingFaceASRBackend",
]

# Optional imports - only available if dependencies are installed
try:
    from .faster_whisper_backend import FasterWhisperBackend
    __all__.append("FasterWhisperBackend")
except ImportError:
    pass

try:
    from .funasr_backend import FunASRBackend
    __all__.append("FunASRBackend")
except ImportError:
    pass

try:
    from .api_backend import OpenAIWhisperBackend, AzureWhisperBackend
    __all__.extend(["OpenAIWhisperBackend", "AzureWhisperBackend"])
except ImportError:
    pass
