"""Example: Using OpenAI Whisper API for dataset creation

This uses the OpenAI API for transcription, which requires an API key
but provides high-quality results without local GPU requirements.
"""

import os
from ovl.data import UniversalDatasetMaker

# Configuration for OpenAI Whisper API
asr_config = {
    "api_key": os.environ.get("OPENAI_API_KEY"),  # Or set directly
    "model": "whisper-1",
    "language": None,  # Auto-detect, or specify: "en", "zh", "ja", etc.
    "temperature": 0,
    "response_format": "verbose_json",  # Get detailed response with timestamps
    "max_retries": 3,
    "retry_delay": 1,
}

# Custom format with specific fields
format_config = {
    "field_mapping": {
        "transcription": "text",  # Map "text" source to "transcription" field
        "audio_path": "audio",  # Map "audio" source to "audio_path" field
        "language": "metadata.language",  # Extract language from metadata
    }
}

# Create dataset maker
maker = UniversalDatasetMaker(
    asr_backend="openai",
    output_format="custom",
    output_dir="data",
    asr_config=asr_config,
    format_config=format_config,
)

# Process dataset (no VAD needed as API handles segmentation)
info = maker.process_dataset(
    input_dir="path/to/audio/files",
    dataset_name="openai_dataset",
    use_vad=False,  # Process files as-is, or set to True for pre-segmentation
    progress_callback=lambda progress, msg: print(f"[{progress*100:.1f}%] {msg}"),
)

print(f"Dataset created: {info['num_samples']} samples")

# Cleanup
maker.cleanup()
