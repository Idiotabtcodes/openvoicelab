# Universal Dataset Maker

A flexible and powerful dataset creation tool that supports multiple ASR backends, customizable output formats, and advanced features like VAD segmentation.

## Features

- **Multiple ASR Backends**: Choose from various speech recognition engines
  - Faster-Whisper (fast and efficient)
  - HuggingFace Transformers (flexible)
  - FunASR (excellent Chinese support)
  - OpenAI Whisper API (cloud-based)
  - Azure Whisper (enterprise)

- **Customizable Output Formats**: Flexible JSONL formatting
  - VibeVoice (default training format)
  - Simple (text + audio)
  - Extended (with metadata)
  - HuggingFace datasets format
  - ESPnet format
  - Custom (define your own)

- **Advanced Processing**:
  - Optional VAD-based audio segmentation
  - Batch transcription support
  - Progress tracking
  - Multilingual support
  - Chinese ASR models

## Quick Start

### Basic Usage

```python
from ovl.data import UniversalDatasetMaker

# Create dataset maker with Faster-Whisper
maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    output_format="simple",
    output_dir="data",
    asr_config={"model_size": "base", "device": "cuda"},
)

# Process audio files
maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="my_dataset",
    use_vad=True,
)

# Cleanup
maker.cleanup()
```

## ASR Backends

### 1. Faster-Whisper (Recommended)

Fast and memory-efficient Whisper implementation using CTranslate2.

```python
asr_config = {
    "model_size": "base",  # tiny, base, small, medium, large-v3
    "device": "cuda",      # cuda, cpu, auto
    "compute_type": "float16",  # float16, int8
    "language": None,      # Auto-detect or specify: "en", "zh", etc.
    "beam_size": 5,
    "vad_filter": True,
}

maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    asr_config=asr_config,
)
```

**Installation**: `pip install faster-whisper`

### 2. HuggingFace Transformers

Standard HuggingFace implementation with broad model support.

```python
asr_config = {
    "model_id": "openai/whisper-base",  # Any HF Whisper model
    "device": "cuda",
    "language": "en",
    "task": "transcribe",  # or "translate"
}

maker = UniversalDatasetMaker(
    asr_backend="huggingface",
    asr_config=asr_config,
)
```

**Installation**: Already included (transformers, torch)

### 3. FunASR (Chinese ASR)

Alibaba's FunASR with excellent Chinese language support.

```python
asr_config = {
    "model_name": "paraformer-zh",  # Chinese ASR
    # Other options:
    # - "paraformer-en": English
    # - "sensevoice": Multilingual (zh, en, ja, ko, yue)
    "device": "cuda",
    "language": "zh",
}

maker = UniversalDatasetMaker(
    asr_backend="funasr",
    asr_config=asr_config,
)
```

**Installation**: `pip install funasr`

### 4. OpenAI Whisper API

Cloud-based transcription using OpenAI's API.

```python
import os

asr_config = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "model": "whisper-1",
    "response_format": "verbose_json",  # Get timestamps
    "language": None,  # Auto-detect
}

maker = UniversalDatasetMaker(
    asr_backend="openai",
    asr_config=asr_config,
)
```

**Installation**: `pip install openai`

### 5. Azure Whisper

Enterprise Azure OpenAI Whisper service.

```python
asr_config = {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_key": "your-azure-key",
    "api_version": "2024-02-01",
    "deployment_name": "whisper",
}

maker = UniversalDatasetMaker(
    asr_backend="azure",
    asr_config=asr_config,
)
```

**Installation**: `pip install openai`

## Output Formats

### VibeVoice Format (Default)

Standard training format for VibeVoice models.

```python
format_config = {
    "speaker_prefix": "Speaker 0: "
}

maker = UniversalDatasetMaker(
    output_format="vibevoice",
    format_config=format_config,
)
```

**Output**: `{"text": "Speaker 0: Hello world", "audio": "path/to/audio.wav"}`

### Simple Format

Minimal format with just text and audio.

```python
maker = UniversalDatasetMaker(output_format="simple")
```

**Output**: `{"text": "Hello world", "audio": "path/to/audio.wav"}`

### Extended Format

Includes metadata like language, confidence, duration.

```python
format_config = {
    "include_language": True,
    "include_confidence": True,
    "include_duration": True,
}

maker = UniversalDatasetMaker(
    output_format="extended",
    format_config=format_config,
)
```

**Output**:
```json
{
  "text": "Hello world",
  "audio": "path/to/audio.wav",
  "language": "en",
  "confidence": 0.95,
  "duration": 2.5
}
```

### HuggingFace Format

Compatible with HuggingFace datasets library.

```python
format_config = {
    "text_column": "sentence",
    "audio_column": "audio",
    "include_language": True,
}

maker = UniversalDatasetMaker(
    output_format="huggingface",
    format_config=format_config,
)
```

**Output**:
```json
{
  "sentence": "Hello world",
  "audio": {"path": "path/to/audio.wav"},
  "language": "en"
}
```

### ESPnet Format

Compatible with ESPnet speech processing toolkit.

```python
maker = UniversalDatasetMaker(output_format="espnet")
```

**Output**:
```json
{
  "utt_id": "audio_001",
  "text": "Hello world",
  "wav": "path/to/audio.wav"
}
```

### Custom Format

Define your own field mapping and transformations.

```python
format_config = {
    "field_mapping": {
        "transcription": "text",
        "file_path": "audio",
        "lang": "metadata.language",
        "model": "metadata.model",
    }
}

maker = UniversalDatasetMaker(
    output_format="custom",
    format_config=format_config,
)
```

**Output**:
```json
{
  "transcription": "Hello world",
  "file_path": "path/to/audio.wav",
  "lang": "en",
  "model": "whisper-base"
}
```

## Advanced Features

### VAD Segmentation

Automatically segment audio files using Voice Activity Detection.

```python
info = maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="my_dataset",
    use_vad=True,
    vad_config={
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
    }
)
```

### Progress Tracking

Monitor processing progress in real-time.

```python
def progress_callback(progress, message):
    print(f"[{progress*100:.1f}%] {message}")

info = maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="my_dataset",
    progress_callback=progress_callback,
)
```

### Batch Processing

Process multiple files efficiently (backend-dependent).

```python
# Automatically uses batch processing if supported by backend
info = maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="my_dataset",
)
```

### Context Manager

Automatic resource cleanup.

```python
with UniversalDatasetMaker(
    asr_backend="faster-whisper",
    asr_config={"model_size": "base"},
) as maker:
    maker.process_dataset(
        input_dir="path/to/audio",
        dataset_name="my_dataset",
    )
# Resources automatically cleaned up
```

## CLI Usage

Process datasets from the command line.

```bash
# Basic usage
python -m ovl.data.cli \
    --input path/to/audio \
    --output my_dataset \
    --backend faster-whisper \
    --format simple

# With configuration file
python -m ovl.data.cli \
    --config config.yaml

# Chinese ASR
python -m ovl.data.cli \
    --input path/to/chinese/audio \
    --output chinese_dataset \
    --backend funasr \
    --asr-config '{"model_name": "paraformer-zh", "language": "zh"}'
```

## Offline Usage and Local Models

All ASR backends are designed to **prefer cached/local models first** for convenient offline usage.

### Automatic Caching

Models are automatically cached after first download:

- **HuggingFace models**: Cached in `~/.cache/huggingface/hub/`
- **Faster-Whisper models**: Cached in `~/.cache/huggingface/hub/`
- **FunASR models**: Cached in `~/.cache/modelscope/hub/`

Once cached, the models work offline automatically without any configuration needed.

### Force Offline Mode

#### HuggingFace Backend

```python
# Automatically tries cached model first, downloads only if needed
maker = UniversalDatasetMaker(
    asr_backend="huggingface",
    asr_config={"model_id": "openai/whisper-base"},
)

# The backend automatically:
# 1. First tries: local_files_only=True (uses cache)
# 2. If not cached: downloads from HuggingFace
```

#### Faster-Whisper Backend

```python
# Use only cached models (fail if not cached)
maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    asr_config={
        "model_size": "base",
        "local_files_only": True,  # Fail if model not cached
    },
)

# Custom cache directory
maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    asr_config={
        "model_size": "base",
        "download_root": "/path/to/custom/cache",
    },
)
```

### Using Local Model Files

You can use local model files instead of downloading:

#### HuggingFace Backend

```python
# Pass local model path instead of model ID
maker = UniversalDatasetMaker(
    asr_backend="huggingface",
    asr_config={
        "model_id": "/path/to/local/whisper-base",
    },
)
```

#### Faster-Whisper Backend

```python
# Pass local model directory as model_size
maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    asr_config={
        "model_size": "/path/to/local/whisper-base-ct2",  # CTranslate2 format
    },
)
```

### Environment Variables for Offline Mode

```bash
# Set HuggingFace offline mode globally
export HF_HUB_OFFLINE=1

# Custom cache directory
export HF_HOME=/path/to/custom/cache

# Then run your script
python my_dataset_script.py
```

### Pre-downloading Models

To ensure offline usage, pre-download models once:

```python
# Pre-download HuggingFace model
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Pre-download Faster-Whisper model
from faster_whisper import WhisperModel
model = WhisperModel("base")

# Models are now cached and will work offline
```

## Complete Examples

See the `examples/dataset_maker_configs/` directory for complete examples:

- `faster_whisper_example.py`: Using Faster-Whisper
- `chinese_asr_example.py`: Chinese dataset with FunASR
- `openai_api_example.py`: Cloud-based with OpenAI API
- `custom_format_example.py`: Custom output format

## Comparison with Original DatasetBuilder

| Feature | DatasetBuilder | UniversalDatasetMaker |
|---------|---------------|----------------------|
| ASR Backends | HF Whisper only | Multiple (Faster-Whisper, FunASR, APIs) |
| Output Format | LJSpeech CSV | Customizable JSONL + CSV |
| Chinese Support | Limited | Excellent (FunASR) |
| API Support | No | Yes (OpenAI, Azure) |
| Format Templates | Fixed | Pluggable |
| Batch Processing | No | Yes (backend-dependent) |
| Offline Mode | Cached models | Cached models (prefers local first) |
| Local Models | Not documented | Full support |

## Installation

### Core Dependencies

```bash
pip install torch torchaudio transformers silero-vad
```

### Optional ASR Backends

```bash
# Faster-Whisper (recommended)
pip install faster-whisper

# FunASR (for Chinese)
pip install funasr

# API backends
pip install openai
```

## Troubleshooting

### ImportError for optional backends

If you see an import error for a specific backend (e.g., `faster-whisper`, `funasr`), install the corresponding package:

```bash
pip install faster-whisper  # for Faster-Whisper
pip install funasr          # for FunASR
pip install openai          # for OpenAI/Azure APIs
```

### CUDA out of memory

Try:
1. Using a smaller model: `model_size="tiny"` or `"base"`
2. Reducing compute precision: `compute_type="int8"`
3. Processing files sequentially instead of batch

### Chinese characters not displaying

Ensure your JSONL files use UTF-8 encoding. The dataset maker automatically handles this.

## License

Same as OpenVoiceLab project license.
