"""Base class for ASR backends"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TranscriptionResult:
    """Result from ASR transcription"""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ASRBackend(ABC):
    """Abstract base class for ASR backends"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ASR backend with configuration

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe a single audio file

        Args:
            audio_path: Path to audio file

        Returns:
            TranscriptionResult with transcription and metadata
        """
        pass

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[TranscriptionResult]:
        """Transcribe multiple audio files

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of TranscriptionResult objects
        """
        # Default implementation: transcribe one by one
        return [self.transcribe(path) for path in audio_paths]

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes

        Returns:
            List of ISO language codes (e.g., ['en', 'zh', 'ja'])
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend name for logging and identification

        Returns:
            Backend name string
        """
        pass

    def cleanup(self):
        """Cleanup resources (models, connections, etc.)"""
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
