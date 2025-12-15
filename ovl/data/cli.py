"""Command-line interface for Universal Dataset Maker"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .universal_dataset_maker import UniversalDatasetMaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_json_config(config_str: str) -> Dict[str, Any]:
    """Parse JSON configuration string"""
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON configuration: {e}")
        sys.exit(1)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Try JSON first
    if config_path.suffix == ".json":
        with open(config_path) as f:
            return json.load(f)

    # Try YAML
    elif config_path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            sys.exit(1)
    else:
        logger.error(f"Unsupported config file format: {config_path.suffix}")
        sys.exit(1)


def progress_callback(progress: float, message: str):
    """Progress callback for CLI"""
    print(f"[{progress*100:5.1f}%] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Dataset Maker - Create speech datasets with flexible ASR backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Faster-Whisper
  %(prog)s --input audio/ --output my_dataset --backend faster-whisper

  # Chinese dataset with FunASR
  %(prog)s --input audio/ --output chinese_dataset --backend funasr \\
    --asr-config '{"model_name": "paraformer-zh", "language": "zh"}'

  # Using configuration file
  %(prog)s --config config.yaml

  # Custom output format
  %(prog)s --input audio/ --output custom_dataset --backend huggingface \\
    --format extended --format-config '{"include_language": true}'

  # OpenAI API without VAD
  %(prog)s --input audio/ --output api_dataset --backend openai --no-vad
        """
    )

    # Configuration file (alternative to CLI args)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON or YAML). Overrides other arguments."
    )

    # Required arguments (unless using --config)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input directory containing audio files"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output dataset name"
    )

    # ASR backend configuration
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "huggingface", "funasr", "openai", "azure"],
        help="ASR backend to use (default: faster-whisper)"
    )

    parser.add_argument(
        "--asr-config",
        type=str,
        help="ASR backend configuration as JSON string"
    )

    # Output format configuration
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="vibevoice",
        choices=["vibevoice", "simple", "extended", "huggingface", "espnet", "custom"],
        help="Output format template (default: vibevoice)"
    )

    parser.add_argument(
        "--format-config",
        type=str,
        help="Format configuration as JSON string"
    )

    # Processing options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for all datasets (default: data)"
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD segmentation (process files as-is)"
    )

    parser.add_argument(
        "--vad-config",
        type=str,
        help="VAD configuration as JSON string"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Load configuration from file if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)

        input_dir = config.get("input_dir")
        dataset_name = config.get("dataset_name")
        backend = config.get("asr_backend", "faster-whisper")
        output_format = config.get("output_format", "vibevoice")
        output_dir = config.get("output_dir", "data")
        asr_config = config.get("asr_config", {})
        format_config = config.get("format_config", {})
        use_vad = config.get("use_vad", True)
        vad_config = config.get("vad_config", {})

    else:
        # Use CLI arguments
        if not args.input or not args.output:
            parser.error("--input and --output are required (unless using --config)")

        input_dir = args.input
        dataset_name = args.output
        backend = args.backend
        output_format = args.format
        output_dir = args.output_dir
        use_vad = not args.no_vad

        # Parse JSON configs
        asr_config = parse_json_config(args.asr_config) if args.asr_config else {}
        format_config = parse_json_config(args.format_config) if args.format_config else {}
        vad_config = parse_json_config(args.vad_config) if args.vad_config else {}

    # Validate input directory
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Create dataset maker
    logger.info(f"Initializing Universal Dataset Maker")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Format: {output_format}")
    logger.info(f"  Output: {output_dir}/{dataset_name}")

    try:
        maker = UniversalDatasetMaker(
            asr_backend=backend,
            output_format=output_format,
            output_dir=output_dir,
            asr_config=asr_config,
            format_config=format_config,
        )
    except Exception as e:
        logger.error(f"Failed to initialize dataset maker: {e}")
        sys.exit(1)

    # Process dataset
    logger.info(f"Processing audio files from {input_dir}")

    try:
        info = maker.process_dataset(
            input_dir=input_dir,
            dataset_name=dataset_name,
            use_vad=use_vad,
            vad_config=vad_config,
            progress_callback=None if args.quiet else progress_callback,
        )

        # Print summary
        print("\n" + "="*60)
        print("Dataset Creation Complete!")
        print("="*60)
        print(f"Dataset name:     {info['name']}")
        print(f"Samples:          {info['num_samples']}")
        print(f"Total duration:   {info['total_duration']:.2f}s ({info['total_duration']/60:.1f} minutes)")
        print(f"ASR backend:      {info['asr_backend']}")
        print(f"Output format:    {info['output_format']}")
        print(f"Location:         {output_dir}/{dataset_name}/")
        print("="*60)

        # List output files
        dataset_dir = Path(output_dir) / dataset_name
        print("\nGenerated files:")
        print(f"  - metadata.jsonl  ({info['num_samples']} entries)")
        if (dataset_dir / "metadata.csv").exists():
            print(f"  - metadata.csv    (LJSpeech format)")
        print(f"  - wavs/           ({info['num_samples']} audio files)")
        print(f"  - info.json       (dataset metadata)")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process dataset: {e}", exc_info=True)
        sys.exit(1)
    finally:
        maker.cleanup()

    logger.info("Done!")


if __name__ == "__main__":
    main()
