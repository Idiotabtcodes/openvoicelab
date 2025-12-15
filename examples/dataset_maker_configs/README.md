# Universal Dataset Maker Examples

This directory contains example configurations and scripts for the Universal Dataset Maker.

## Quick Start

### 1. Basic Usage - Faster-Whisper

```python
from ovl.data import UniversalDatasetMaker

maker = UniversalDatasetMaker(
    asr_backend="faster-whisper",
    output_format="simple",
    asr_config={"model_size": "base", "device": "cuda"},
)

maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="my_dataset",
    use_vad=True,
)
```

### 2. Chinese ASR with FunASR

```python
from ovl.data import UniversalDatasetMaker

maker = UniversalDatasetMaker(
    asr_backend="funasr",
    asr_config={
        "model_name": "paraformer-zh",
        "language": "zh",
    },
)

maker.process_dataset(
    input_dir="path/to/chinese/audio",
    dataset_name="chinese_dataset",
)
```

### 3. Using OpenAI API

```python
import os
from ovl.data import UniversalDatasetMaker

maker = UniversalDatasetMaker(
    asr_backend="openai",
    asr_config={
        "api_key": os.environ["OPENAI_API_KEY"],
    },
)

maker.process_dataset(
    input_dir="path/to/audio",
    dataset_name="openai_dataset",
    use_vad=False,  # API handles segmentation
)
```

## Example Files

- **`faster_whisper_example.py`**: Complete example using Faster-Whisper backend
- **`chinese_asr_example.py`**: Chinese dataset creation with FunASR
- **`openai_api_example.py`**: Cloud-based transcription with OpenAI API
- **`multilingual_example.py`**: Multilingual dataset with SenseVoice
- **`config_example.yaml`**: YAML configuration file template

## CLI Usage

Process datasets from the command line:

```bash
# Basic usage
python -m ovl.data.cli --input audio/ --output my_dataset --backend faster-whisper

# With configuration file
python -m ovl.data.cli --config config_example.yaml

# Chinese ASR
python -m ovl.data.cli --input audio/ --output chinese_dataset --backend funasr \
  --asr-config '{"model_name": "paraformer-zh", "language": "zh"}'
```

## Installation

### Core dependencies (required)
```bash
pip install torch torchaudio transformers silero-vad
```

### Optional backends

```bash
# Faster-Whisper (recommended for speed)
pip install faster-whisper

# FunASR (for Chinese)
pip install funasr

# API support
pip install openai
```

## Output Formats

The Universal Dataset Maker supports multiple output formats:

1. **vibevoice** (default): `{"text": "Speaker 0: ...", "audio": "..."}`
2. **simple**: `{"text": "...", "audio": "..."}`
3. **extended**: Includes metadata like language, confidence
4. **huggingface**: Compatible with HuggingFace datasets
5. **espnet**: ESPnet toolkit format
6. **custom**: Define your own field mapping

## Documentation

See the full documentation at: `docs/UNIVERSAL_DATASET_MAKER.md`

## Comparison with Original DatasetBuilder

| Feature | DatasetBuilder | UniversalDatasetMaker |
|---------|---------------|----------------------|
| ASR Backends | HF Whisper only | Multiple (Faster-Whisper, FunASR, APIs) |
| Output Format | Fixed LJSpeech | Customizable JSONL + CSV |
| Chinese Support | Limited | Excellent (FunASR) |
| API Support | No | Yes (OpenAI, Azure) |

## Support

For issues and questions:
- Check the documentation: `docs/UNIVERSAL_DATASET_MAKER.md`
- Review example scripts in this directory
- Open an issue on GitHub
