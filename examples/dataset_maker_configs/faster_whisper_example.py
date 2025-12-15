"""Example: Using Faster-Whisper backend for dataset creation

Faster-Whisper is significantly faster and more memory-efficient than
the original OpenAI Whisper implementation.
"""

from ovl.data import UniversalDatasetMaker

# Configuration for Faster-Whisper backend
asr_config = {
    "model_size": "base",  # Options: tiny, base, small, medium, large-v3
    "device": "cuda",  # Options: cuda, cpu, auto
    "compute_type": "float16",  # Options: float16, int8, int8_float16
    "language": None,  # Auto-detect language, or specify: "en", "zh", "ja", etc.
    "beam_size": 5,
    "vad_filter": True,  # Enable VAD filtering for cleaner transcriptions
}

# Simple output format
format_config = {}

# Create dataset maker
maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    output_format="simple",  # Options: vibevoice, simple, extended, huggingface, espnet
    output_dir="data",
    asr_config=asr_config,
    format_config=format_config,
)

# Process dataset
info = maker.process_dataset(
    input_dir="path/to/your/audio/files",
    dataset_name="my_dataset_faster_whisper",
    use_vad=True,  # Use VAD for segmentation
    progress_callback=lambda progress, msg: print(f"[{progress*100:.1f}%] {msg}"),
)

print(f"Dataset created: {info['num_samples']} samples, {info['total_duration']:.2f}s")

# Cleanup
maker.cleanup()
