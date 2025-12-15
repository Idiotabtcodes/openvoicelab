"""FunASR backend for Chinese and multilingual ASR"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from funasr import AutoModel

from .base import ASRBackend, TranscriptionResult


class FunASRBackend(ASRBackend):
    """ASR backend using FunASR (Alibaba's speech recognition toolkit)

    FunASR provides excellent Chinese ASR support and various pretrained models.
    Popular models:
    - paraformer-zh: Chinese ASR
    - paraformer-en: English ASR
    - SenseVoiceSmall: Multilingual (Chinese, English, Japanese, Korean, Cantonese)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Get configuration
        self.model_name = self.config.get("model_name", "paraformer-zh")
        self.device = self.config.get("device", "cuda")
        self.batch_size = self.config.get("batch_size", 1)
        self.language = self.config.get("language", "zh")  # zh, en, ja, ko, yue (Cantonese)

        # Model mappings for common names
        model_map = {
            "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "paraformer-en": "iic/speech_paraformer-large_asr_nat-en-16k-common-vocab10020",
            "sensevoice": "iic/SenseVoiceSmall",
            "whisper-large-v3": "FunAudioLLM/SenseVoiceSmall",  # Also supports Whisper
        }

        # Resolve model name
        resolved_model = model_map.get(self.model_name, self.model_name)

        # Initialize model
        self.model = AutoModel(
            model=resolved_model,
            device=self.device,
        )

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio using FunASR"""

        # Perform transcription
        result = self.model.generate(
            input=str(audio_path),
            batch_size=self.batch_size,
        )

        # Extract text from result
        # FunASR returns a list of dictionaries
        if isinstance(result, list) and len(result) > 0:
            item = result[0]

            if isinstance(item, dict):
                text = item.get("text", "").strip()
                # FunASR can provide timestamps in some models
                segments = item.get("timestamp", None)
            else:
                text = str(item).strip()
                segments = None
        else:
            text = str(result).strip() if result else ""
            segments = None

        return TranscriptionResult(
            text=text,
            language=self.language,
            segments=segments,
            metadata={
                "model": self.model_name,
                "device": self.device,
            }
        )

    def transcribe_batch(self, audio_paths: List[Union[str, Path]]) -> List[TranscriptionResult]:
        """Transcribe multiple files in batch (more efficient)"""

        # FunASR supports batch processing
        audio_paths_str = [str(p) for p in audio_paths]

        results = self.model.generate(
            input=audio_paths_str,
            batch_size=self.batch_size,
        )

        transcriptions = []
        for result_item in results:
            if isinstance(result_item, dict):
                text = result_item.get("text", "").strip()
                segments = result_item.get("timestamp", None)
            else:
                text = str(result_item).strip()
                segments = None

            transcriptions.append(TranscriptionResult(
                text=text,
                language=self.language,
                segments=segments,
                metadata={
                    "model": self.model_name,
                    "device": self.device,
                }
            ))

        return transcriptions

    def get_supported_languages(self) -> List[str]:
        """Get supported languages based on model"""
        # Model-specific language support
        model_languages = {
            "paraformer-zh": ["zh"],
            "paraformer-en": ["en"],
            "sensevoice": ["zh", "en", "ja", "ko", "yue"],  # Multilingual
        }

        return model_languages.get(self.model_name, ["zh"])

    def get_backend_name(self) -> str:
        return f"FunASR-{self.model_name}"

    def cleanup(self):
        """Cleanup model resources"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
