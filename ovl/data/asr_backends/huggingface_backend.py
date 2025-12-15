"""HuggingFace Transformers ASR backend"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import pipeline as hf_pipeline

from .base import ASRBackend, TranscriptionResult


class HuggingFaceASRBackend(ASRBackend):
    """ASR backend using HuggingFace Transformers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Get configuration
        self.model_id = self.config.get("model_id", "openai/whisper-base")
        self.device = self.config.get("device", self._auto_detect_device())
        self.chunk_length_s = self.config.get("chunk_length_s", 30)
        self.return_timestamps = self.config.get("return_timestamps", True)
        self.language = self.config.get("language")  # Optional: force language
        self.task = self.config.get("task", "transcribe")  # or "translate"

        # Initialize pipeline
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language
        if self.task:
            generate_kwargs["task"] = self.task

        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            chunk_length_s=self.chunk_length_s,
            return_timestamps=self.return_timestamps,
            generate_kwargs=generate_kwargs if generate_kwargs else None,
        )

    def _auto_detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio using HuggingFace pipeline"""
        result = self.pipe(str(audio_path))

        # Extract text and metadata
        if isinstance(result, dict):
            text = result.get("text", "").strip()

            # Extract segments if available
            segments = None
            if "chunks" in result:
                segments = [
                    {
                        "text": chunk.get("text", "").strip(),
                        "timestamp": chunk.get("timestamp", [None, None]),
                    }
                    for chunk in result["chunks"]
                ]
        else:
            text = str(result).strip()
            segments = None

        return TranscriptionResult(
            text=text,
            language=self.language,
            segments=segments,
            metadata={
                "model": self.model_id,
                "device": self.device,
            }
        )

    def get_supported_languages(self) -> List[str]:
        """Whisper supports 99 languages"""
        # Major languages supported by Whisper
        return [
            "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "tr", "vi", "th", "id", "nl", "pl", "uk", "ro",
            # ... Whisper supports 99 languages total
        ]

    def get_backend_name(self) -> str:
        return f"HuggingFace-{self.model_id}"

    def cleanup(self):
        """Cleanup model resources"""
        if hasattr(self, "pipe") and self.pipe is not None:
            # Move model to CPU to free GPU memory
            if hasattr(self.pipe, "model"):
                self.pipe.model.to("cpu")
            del self.pipe
            self.pipe = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
