"""Example: Multilingual dataset creation with SenseVoice

SenseVoice supports Chinese, English, Japanese, Korean, and Cantonese.
"""

from ovl.data import UniversalDatasetMaker

# Configuration for FunASR SenseVoice (multilingual)
asr_config = {
    "model_name": "sensevoice",  # Multilingual model
    "device": "cuda",
    "language": "auto",  # Auto-detect language, or specify: "zh", "en", "ja", "ko", "yue"
    "batch_size": 4,  # Process multiple files at once
}

# Extended format with language detection
format_config = {
    "include_language": True,  # Include detected language in output
    "include_confidence": False,
    "include_duration": True,
}

# Create dataset maker
maker = UniversalDatasetMaker(
    asr_backend="funasr",
    output_format="extended",
    output_dir="data",
    asr_config=asr_config,
    format_config=format_config,
)

# Process multilingual audio dataset
info = maker.process_dataset(
    input_dir="path/to/multilingual/audio",
    dataset_name="multilingual_dataset",
    use_vad=True,
    progress_callback=lambda progress, msg: print(f"[{progress*100:.1f}%] {msg}"),
)

print(f"\nDataset created: {info['num_samples']} samples")
print(f"Total duration: {info['total_duration']:.2f}s")

# List datasets
datasets = maker.list_datasets()
print(f"\nAll datasets: {len(datasets)}")
for ds in datasets:
    print(f"  - {ds['name']}: {ds['num_samples']} samples, {ds.get('asr_backend', 'N/A')}")

# Cleanup
maker.cleanup()
