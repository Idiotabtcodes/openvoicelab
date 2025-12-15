"""Example: Using FunASR for Chinese dataset creation

FunASR provides excellent Chinese ASR support with models like Paraformer.
"""

from ovl.data import UniversalDatasetMaker

# Configuration for FunASR backend
asr_config = {
    "model_name": "paraformer-zh",  # Chinese ASR
    # Other options:
    # - "paraformer-en": English ASR
    # - "sensevoice": Multilingual (Chinese, English, Japanese, Korean, Cantonese)
    "device": "cuda",
    "language": "zh",
    "batch_size": 1,
}

# Extended format with metadata
format_config = {
    "include_language": True,
    "include_confidence": False,
    "include_duration": False,
}

# Create dataset maker
maker = UniversalDatasetMaker(
    asr_backend="funasr",
    output_format="extended",
    output_dir="data",
    asr_config=asr_config,
    format_config=format_config,
)

# Process Chinese audio dataset
info = maker.process_dataset(
    input_dir="path/to/chinese/audio/files",
    dataset_name="chinese_dataset",
    use_vad=True,
    vad_config={
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
    },
    progress_callback=lambda progress, msg: print(f"[{progress*100:.1f}%] {msg}"),
)

print(f"Chinese dataset created: {info['num_samples']} samples")

# Cleanup
maker.cleanup()
