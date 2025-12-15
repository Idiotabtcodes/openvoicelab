"""API-based ASR backends (OpenAI, Azure, etc.)"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base import ASRBackend, TranscriptionResult


class OpenAIWhisperBackend(ASRBackend):
    """ASR backend using OpenAI Whisper API

    Requires: openai>=1.0.0
    API Key: Set via OPENAI_API_KEY environment variable or pass in config
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")

        super().__init__(config)

        # Get configuration
        api_key = self.config.get("api_key")  # Falls back to OPENAI_API_KEY env var
        self.model = self.config.get("model", "whisper-1")
        self.language = self.config.get("language")  # Optional: force language (ISO-639-1)
        self.temperature = self.config.get("temperature", 0)
        self.response_format = self.config.get("response_format", "verbose_json")  # text, json, verbose_json
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)

        # Initialize client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper API"""

        audio_path = Path(audio_path)

        # Prepare request parameters
        kwargs = {
            "model": self.model,
            "response_format": self.response_format,
            "temperature": self.temperature,
        }

        if self.language:
            kwargs["language"] = self.language

        # Retry logic for API calls
        for attempt in range(self.max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        **kwargs
                    )

                # Parse response based on format
                if self.response_format == "verbose_json":
                    text = response.text
                    language = getattr(response, "language", self.language)
                    segments = getattr(response, "segments", None)

                    # Convert segments to our format
                    formatted_segments = None
                    if segments:
                        formatted_segments = [
                            {
                                "text": seg.get("text", "").strip(),
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                            }
                            for seg in segments
                        ]

                    return TranscriptionResult(
                        text=text,
                        language=language,
                        segments=formatted_segments,
                        metadata={
                            "model": self.model,
                            "duration": getattr(response, "duration", None),
                        }
                    )
                elif self.response_format == "json":
                    return TranscriptionResult(
                        text=response.text,
                        language=self.language,
                        metadata={"model": self.model}
                    )
                else:  # text format
                    return TranscriptionResult(
                        text=response,
                        language=self.language,
                        metadata={"model": self.model}
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Failed to transcribe {audio_path} after {self.max_retries} attempts: {e}")

    def get_supported_languages(self) -> List[str]:
        """OpenAI Whisper supports 99 languages"""
        return [
            "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "tr", "vi", "th", "id", "nl", "pl", "uk", "ro",
            # ... 99 languages total
        ]

    def get_backend_name(self) -> str:
        return f"OpenAI-{self.model}"


class AzureWhisperBackend(ASRBackend):
    """ASR backend using Azure OpenAI Whisper

    Requires: openai>=1.0.0
    Configuration:
    - endpoint: Azure OpenAI endpoint URL
    - api_key: Azure API key
    - api_version: API version (default: 2024-02-01)
    - deployment_name: Deployment name for Whisper model
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")

        super().__init__(config)

        # Get configuration
        endpoint = self.config.get("endpoint")
        api_key = self.config.get("api_key")
        api_version = self.config.get("api_version", "2024-02-01")
        self.deployment_name = self.config.get("deployment_name", "whisper")
        self.language = self.config.get("language")
        self.temperature = self.config.get("temperature", 0)
        self.response_format = self.config.get("response_format", "verbose_json")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1)

        if not endpoint or not api_key:
            raise ValueError("Azure endpoint and api_key are required")

        # Initialize client
        self.client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def transcribe(self, audio_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio using Azure OpenAI Whisper"""

        audio_path = Path(audio_path)

        # Prepare request parameters
        kwargs = {
            "model": self.deployment_name,
            "response_format": self.response_format,
            "temperature": self.temperature,
        }

        if self.language:
            kwargs["language"] = self.language

        # Retry logic for API calls
        for attempt in range(self.max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=audio_file,
                        **kwargs
                    )

                # Parse response (similar to OpenAI)
                if self.response_format == "verbose_json":
                    text = response.text
                    language = getattr(response, "language", self.language)
                    segments = getattr(response, "segments", None)

                    formatted_segments = None
                    if segments:
                        formatted_segments = [
                            {
                                "text": seg.get("text", "").strip(),
                                "start": seg.get("start"),
                                "end": seg.get("end"),
                            }
                            for seg in segments
                        ]

                    return TranscriptionResult(
                        text=text,
                        language=language,
                        segments=formatted_segments,
                        metadata={
                            "model": self.deployment_name,
                            "duration": getattr(response, "duration", None),
                        }
                    )
                elif self.response_format == "json":
                    return TranscriptionResult(
                        text=response.text,
                        language=self.language,
                        metadata={"model": self.deployment_name}
                    )
                else:  # text format
                    return TranscriptionResult(
                        text=response,
                        language=self.language,
                        metadata={"model": self.deployment_name}
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise Exception(f"Failed to transcribe {audio_path} after {self.max_retries} attempts: {e}")

    def get_supported_languages(self) -> List[str]:
        """Azure Whisper supports same languages as OpenAI"""
        return [
            "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "tr", "vi", "th", "id", "nl", "pl", "uk", "ro",
        ]

    def get_backend_name(self) -> str:
        return f"Azure-{self.deployment_name}"
