"""Faster-Whisper ASR backend (CTranslate2-based)"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faster_whisper import WhisperModel

from .base import ASRBackend, TranscriptionResult


class FasterWhisperBackend(ASRBackend):
    """ASR backend using Faster-Whisper (CTranslate2)

    Faster-Whisper is significantly faster and more memory-efficient than
    the original OpenAI Whisper implementation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Get configuration
        self.model_size = self.config.get("model_size", "base")  # tiny, base, small, medium, large-v3
        self.device = self.config.get("device", "auto")  # auto, cuda, cpu
        self.compute_type = self.config.get("compute_type", "default")  # default, int8, float16, int8_float16
        self.language = self.config.get("language")  # Optional: force language
        self.beam_size = self.config.get("beam_size", 5)
        self.vad_filter = self.config.get("vad_filter", True)  # Enable VAD filtering
        self.vad_parameters = self.config.get("vad_parameters")
        self.task = self.config.get("task", "transcribe")  # or "translate"

        # Initialize model
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio using Faster-Whisper"""

        # Transcribe with segments
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        # Collect all segments
        all_segments = []
        full_text = []

        for segment in segments:
            all_segments.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "confidence": getattr(segment, "avg_logprob", None),
            })
            full_text.append(segment.text.strip())

        text = " ".join(full_text)

        # Calculate average confidence
        confidences = [s["confidence"] for s in all_segments if s["confidence"] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return TranscriptionResult(
            text=text,
            language=info.language if hasattr(info, "language") else self.language,
            confidence=avg_confidence,
            segments=all_segments,
            metadata={
                "model": self.model_size,
                "device": self.device,
                "language_probability": getattr(info, "language_probability", None),
                "duration": getattr(info, "duration", None),
            }
        )

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[TranscriptionResult]:
        """Transcribe multiple files efficiently"""
        # Faster-Whisper doesn't have built-in batching, but we can still process sequentially
        # The model is already optimized for speed
        return [self.transcribe(path) for path in audio_paths]

    def get_supported_languages(self) -> List[str]:
        """Whisper supports 99 languages"""
        return [
            "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "tr", "vi", "th", "id", "nl", "pl", "uk", "ro",
            # ... Whisper supports 99 languages total
        ]

    def get_backend_name(self) -> str:
        return f"FasterWhisper-{self.model_size}"

    def cleanup(self):
        """Cleanup model resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
