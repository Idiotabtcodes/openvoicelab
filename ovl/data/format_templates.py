"""Customizable output format templates for dataset generation"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional


class FormatTemplate:
    """Base class for output format templates"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format a single dataset entry

        Args:
            audio_path: Path to audio file
            transcription: Transcription text
            metadata: Optional metadata (language, confidence, segments, etc.)

        Returns:
            Dictionary ready for JSONL output
        """
        raise NotImplementedError


class VibeVoiceTemplate(FormatTemplate):
    """Default VibeVoice training format (existing format)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.speaker_prefix = self.config.get("speaker_prefix", "Speaker 0: ")

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format: {"text": "Speaker 0: <text>", "audio": "<path>"}"""
        return {
            "text": f"{self.speaker_prefix}{transcription}",
            "audio": audio_path
        }


class SimpleTemplate(FormatTemplate):
    """Simple format with just text and audio"""

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format: {"text": "<text>", "audio": "<path>"}"""
        return {
            "text": transcription,
            "audio": audio_path
        }


class ExtendedTemplate(FormatTemplate):
    """Extended format with additional metadata fields"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.include_language = self.config.get("include_language", True)
        self.include_confidence = self.config.get("include_confidence", False)
        self.include_duration = self.config.get("include_duration", False)

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format with metadata: {"text": "<text>", "audio": "<path>", "language": "en", ...}"""
        metadata = metadata or {}

        entry = {
            "text": transcription,
            "audio": audio_path,
        }

        if self.include_language and metadata.get("language"):
            entry["language"] = metadata["language"]

        if self.include_confidence and metadata.get("confidence"):
            entry["confidence"] = metadata["confidence"]

        if self.include_duration and metadata.get("duration"):
            entry["duration"] = metadata["duration"]

        return entry


class HuggingFaceTemplate(FormatTemplate):
    """HuggingFace datasets format"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.text_column = self.config.get("text_column", "sentence")
        self.audio_column = self.config.get("audio_column", "audio")
        self.include_language = self.config.get("include_language", False)

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format: {"sentence": "<text>", "audio": {"path": "<path>"}, ...}"""
        metadata = metadata or {}

        entry = {
            self.text_column: transcription,
            self.audio_column: {"path": audio_path},
        }

        if self.include_language and metadata.get("language"):
            entry["language"] = metadata["language"]

        return entry


class ESPnetTemplate(FormatTemplate):
    """ESPnet toolkit format"""

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format: {"utt_id": "<filename>", "text": "<text>", "wav": "<path>"}"""
        # Extract filename without extension as utterance ID
        utt_id = Path(audio_path).stem

        return {
            "utt_id": utt_id,
            "text": transcription,
            "wav": audio_path,
        }


class CustomTemplate(FormatTemplate):
    """Custom template with user-defined fields and formatters"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Field mapping: output_field -> (source, formatter)
        # source can be: "audio", "text", "metadata.key"
        # formatter is optional callable to transform the value
        self.field_mapping = self.config.get("field_mapping", {})

    def format_entry(
        self,
        audio_path: str,
        transcription: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format with custom field mapping"""
        metadata = metadata or {}

        entry = {}

        for output_field, config in self.field_mapping.items():
            if isinstance(config, str):
                # Simple mapping: "audio", "text", or "metadata.language"
                value = self._get_value(config, audio_path, transcription, metadata)
            elif isinstance(config, dict):
                # With formatter: {"source": "text", "formatter": "upper"}
                source = config.get("source")
                formatter = config.get("formatter")

                value = self._get_value(source, audio_path, transcription, metadata)

                if formatter and callable(formatter):
                    value = formatter(value)
                elif formatter == "upper":
                    value = str(value).upper()
                elif formatter == "lower":
                    value = str(value).lower()
                elif formatter == "strip":
                    value = str(value).strip()
            else:
                continue

            if value is not None:
                entry[output_field] = value

        return entry

    def _get_value(self, source: str, audio_path: str, transcription: str, metadata: Dict[str, Any]) -> Any:
        """Get value from source"""
        if source == "audio":
            return audio_path
        elif source == "text":
            return transcription
        elif source.startswith("metadata."):
            key = source.split(".", 1)[1]
            return metadata.get(key)
        else:
            return None


# Template registry
TEMPLATE_REGISTRY = {
    "vibevoice": VibeVoiceTemplate,
    "simple": SimpleTemplate,
    "extended": ExtendedTemplate,
    "huggingface": HuggingFaceTemplate,
    "espnet": ESPnetTemplate,
    "custom": CustomTemplate,
}


def get_template(name: str, config: Optional[Dict[str, Any]] = None) -> FormatTemplate:
    """Get format template by name

    Args:
        name: Template name (vibevoice, simple, extended, huggingface, espnet, custom)
        config: Optional configuration for the template

    Returns:
        FormatTemplate instance
    """
    template_class = TEMPLATE_REGISTRY.get(name.lower())
    if not template_class:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATE_REGISTRY.keys())}")

    return template_class(config)
