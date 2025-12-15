"""Universal Dataset Maker with flexible ASR backends and output formats"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from .asr_backends.base import ASRBackend
from .format_templates import FormatTemplate, get_template

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


class UniversalDatasetMaker:
    """Universal dataset maker with pluggable ASR backends and output formats

    Features:
    - Multiple ASR backends: Faster-Whisper, HuggingFace, FunASR, OpenAI API, Azure
    - Customizable output formats: VibeVoice, Simple, Extended, HuggingFace, ESPnet, Custom
    - Flexible configuration via dictionary or YAML
    - Supports Chinese and multilingual ASR models
    - Optional VAD-based segmentation
    - Progress callbacks for UI integration
    """

    def __init__(
        self,
        asr_backend: Union[str, ASRBackend],
        output_format: Union[str, FormatTemplate] = "vibevoice",
        output_dir: str = "data",
        asr_config: Optional[Dict[str, Any]] = None,
        format_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize UniversalDatasetMaker

        Args:
            asr_backend: ASR backend name or instance
                Names: "faster-whisper", "huggingface", "funasr", "openai", "azure"
            output_format: Output format name or template instance
                Names: "vibevoice", "simple", "extended", "huggingface", "espnet", "custom"
            output_dir: Output directory for datasets
            asr_config: Configuration dict for ASR backend
            format_config: Configuration dict for output format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ASR backend
        if isinstance(asr_backend, str):
            self.asr_backend = self._create_asr_backend(asr_backend, asr_config or {})
        else:
            self.asr_backend = asr_backend

        # Initialize output format
        if isinstance(output_format, str):
            self.output_format = get_template(output_format, format_config or {})
        else:
            self.output_format = output_format

        # State
        self.processing = False
        self.current_job = None
        self.progress_callback = None

        logger.info(f"Initialized UniversalDatasetMaker with {self.asr_backend.get_backend_name()} backend")

    def _create_asr_backend(self, backend_name: str, config: Dict[str, Any]) -> ASRBackend:
        """Create ASR backend from name and config"""
        backend_name = backend_name.lower()

        if backend_name == "faster-whisper" or backend_name == "faster_whisper":
            from .asr_backends.faster_whisper_backend import FasterWhisperBackend
            return FasterWhisperBackend(config)

        elif backend_name == "huggingface" or backend_name == "hf":
            from .asr_backends.huggingface_backend import HuggingFaceASRBackend
            return HuggingFaceASRBackend(config)

        elif backend_name == "funasr":
            from .asr_backends.funasr_backend import FunASRBackend
            return FunASRBackend(config)

        elif backend_name == "openai":
            from .asr_backends.api_backend import OpenAIWhisperBackend
            return OpenAIWhisperBackend(config)

        elif backend_name == "azure":
            from .asr_backends.api_backend import AzureWhisperBackend
            return AzureWhisperBackend(config)

        else:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: faster-whisper, huggingface, funasr, openai, azure"
            )

    def _iter_audio_files(self, input_dir: Path) -> List[Path]:
        """Get all audio files from input directory"""
        files = []
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                files.append(path)
        return files

    def segment_audio(
        self,
        input_dir: Path,
        segments_dir: Path,
        vad_sampling_rate: int = 16_000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Path]:
        """Split audio files into voice segments using Silero VAD

        Args:
            input_dir: Directory containing input audio files
            segments_dir: Directory to save segmented audio
            vad_sampling_rate: Sampling rate for VAD (16000 Hz recommended)
            min_speech_duration_ms: Minimum speech segment duration in milliseconds
            min_silence_duration_ms: Minimum silence duration between segments in milliseconds

        Returns:
            List of paths to segmented audio files
        """
        segments_dir.mkdir(parents=True, exist_ok=True)
        model = load_silero_vad()
        created_segments: List[Path] = []

        audio_files = self._iter_audio_files(input_dir)
        total_files = len(audio_files)

        logger.info(f"Segmenting {total_files} audio files...")

        for idx, audio_path in enumerate(audio_files):
            if self.progress_callback:
                progress = idx / total_files if total_files > 0 else 0
                self.progress_callback(progress, f"Segmenting audio ({idx+1}/{total_files})")

            try:
                wav = read_audio(str(audio_path), sampling_rate=vad_sampling_rate)
                timestamps = get_speech_timestamps(
                    wav,
                    model,
                    sampling_rate=vad_sampling_rate,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                )

                if not timestamps:
                    logger.warning(f"No speech detected in {audio_path.name}")
                    continue

                audio_tensor, sample_rate = torchaudio.load(str(audio_path))

                for seg_idx, segment in enumerate(timestamps, start=1):
                    start = int(segment["start"] * sample_rate / vad_sampling_rate)
                    end = int(segment["end"] * sample_rate / vad_sampling_rate)

                    if end <= start:
                        continue

                    segment_tensor = audio_tensor[:, start:end]
                    output_path = segments_dir / f"{audio_path.stem}_{seg_idx:03d}.wav"
                    torchaudio.save(str(output_path), segment_tensor, sample_rate)
                    created_segments.append(output_path)

            except Exception as e:
                logger.error(f"Failed to segment {audio_path.name}: {e}")
                continue

        logger.info(f"Created {len(created_segments)} segments from {total_files} files")
        return created_segments

    def transcribe_segments(
        self,
        audio_paths: List[Path],
        use_batch: bool = True,
    ) -> Dict[Path, Dict[str, Any]]:
        """Transcribe audio segments using configured ASR backend

        Args:
            audio_paths: List of audio file paths
            use_batch: Use batch transcription if backend supports it

        Returns:
            Dictionary mapping audio paths to transcription results
        """
        if not audio_paths:
            return {}

        logger.info(f"Transcribing {len(audio_paths)} segments using {self.asr_backend.get_backend_name()}...")

        results = {}

        # Try batch transcription if supported
        if use_batch and hasattr(self.asr_backend, "transcribe_batch"):
            try:
                transcription_results = self.asr_backend.transcribe_batch(audio_paths)

                for path, result in zip(audio_paths, transcription_results):
                    results[path] = {
                        "text": result.text,
                        "language": result.language,
                        "confidence": result.confidence,
                        "segments": result.segments,
                        "metadata": result.metadata,
                    }

                return results

            except Exception as e:
                logger.warning(f"Batch transcription failed, falling back to sequential: {e}")

        # Sequential transcription
        total = len(audio_paths)
        for idx, path in enumerate(audio_paths):
            if self.progress_callback:
                progress = idx / total if total > 0 else 0
                self.progress_callback(progress, f"Transcribing ({idx+1}/{total})")

            try:
                result = self.asr_backend.transcribe(path)
                results[path] = {
                    "text": result.text,
                    "language": result.language,
                    "confidence": result.confidence,
                    "segments": result.segments,
                    "metadata": result.metadata,
                }
            except Exception as e:
                logger.error(f"Failed to transcribe {path.name}: {e}")
                continue

        logger.info(f"Successfully transcribed {len(results)}/{len(audio_paths)} segments")
        return results

    def save_dataset(
        self,
        transcriptions: Dict[Path, Dict[str, Any]],
        dataset_name: str,
        dataset_dir: Path,
        include_csv: bool = True,
    ):
        """Save dataset in configured format

        Args:
            transcriptions: Dictionary of transcription results
            dataset_name: Name of the dataset
            dataset_dir: Output directory
            include_csv: Also save LJSpeech-style metadata.csv for compatibility
        """
        # Create wavs directory
        wavs_dir = dataset_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

        # Prepare JSONL entries
        jsonl_entries = []
        csv_lines = []

        logger.info(f"Saving {len(transcriptions)} samples to {dataset_dir}")

        for idx, (audio_path, trans_data) in enumerate(sorted(transcriptions.items(), key=lambda x: str(x[0]))):
            # Copy wav file to wavs directory
            new_name = f"{dataset_name}_{idx:06d}.wav"
            dest_path = wavs_dir / new_name

            import shutil
            shutil.copy(audio_path, dest_path)

            # Format entry using template
            entry = self.output_format.format_entry(
                audio_path=str(dest_path),
                transcription=trans_data["text"],
                metadata=trans_data.get("metadata", {}),
            )

            jsonl_entries.append(entry)

            # Also save CSV format for compatibility
            if include_csv:
                text = trans_data["text"]
                csv_lines.append(f"{new_name}|{text}|{text}")

        # Write JSONL
        jsonl_path = dataset_dir / "metadata.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"Saved JSONL to {jsonl_path}")

        # Write CSV if requested
        if include_csv:
            csv_path = dataset_dir / "metadata.csv"
            csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
            logger.info(f"Saved CSV to {csv_path}")

    def process_dataset(
        self,
        input_dir: str,
        dataset_name: str,
        use_vad: bool = True,
        vad_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Process audio files into dataset

        Args:
            input_dir: Directory containing input audio files
            dataset_name: Name for the output dataset
            use_vad: Use VAD for audio segmentation
            vad_config: VAD configuration (min_speech_duration_ms, min_silence_duration_ms)
            progress_callback: Callback function for progress updates (progress, message)

        Returns:
            Dictionary with dataset info
        """
        self.processing = True
        self.progress_callback = progress_callback

        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise ValueError(f"Input directory not found: {input_dir}")

            # Create dataset directory
            dataset_dir = self.output_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Get audio files to process
            if use_vad:
                # Step 1: VAD Segmentation
                if progress_callback:
                    progress_callback(0, "Starting VAD segmentation...")

                work_dir = dataset_dir / "_work"
                segments_dir = work_dir / "segments"

                vad_config = vad_config or {}
                segments = self.segment_audio(input_path, segments_dir, **vad_config)

                if not segments:
                    raise ValueError("No audio segments found after VAD processing")

                audio_to_transcribe = segments
            else:
                # No VAD: transcribe files directly
                audio_to_transcribe = self._iter_audio_files(input_path)

                if not audio_to_transcribe:
                    raise ValueError(f"No audio files found in {input_dir}")

            # Step 2: Transcription
            if progress_callback:
                progress_callback(0.3, "Starting transcription...")

            transcriptions = self.transcribe_segments(audio_to_transcribe)

            if not transcriptions:
                raise ValueError("No successful transcriptions")

            # Step 3: Save dataset
            if progress_callback:
                progress_callback(0.9, "Saving dataset...")

            self.save_dataset(transcriptions, dataset_name, dataset_dir)

            # Calculate statistics
            total_duration = self._calculate_dataset_duration(dataset_dir)

            # Save dataset info
            info = {
                "name": dataset_name,
                "created_at": datetime.now().isoformat(),
                "num_samples": len(transcriptions),
                "total_duration": total_duration,
                "asr_backend": self.asr_backend.get_backend_name(),
                "output_format": type(self.output_format).__name__,
                "input_dir": str(input_dir),
                "used_vad": use_vad,
            }

            info_path = dataset_dir / "info.json"
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

            logger.info(f"Dataset created: {len(transcriptions)} samples, {total_duration:.2f}s total duration")

            if progress_callback:
                progress_callback(1.0, f"Dataset created: {len(transcriptions)} samples")

            return info

        finally:
            self.processing = False
            self.progress_callback = None

    def process_dataset_async(self, *args, **kwargs):
        """Process dataset in background thread"""
        thread = threading.Thread(target=self.process_dataset, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    def _calculate_dataset_duration(self, dataset_dir: Path) -> float:
        """Calculate total duration of all audio files in dataset (in seconds)"""
        wavs_dir = dataset_dir / "wavs"
        if not wavs_dir.exists():
            return 0.0

        total_duration = 0.0
        for wav_file in wavs_dir.glob("*.wav"):
            try:
                metadata = torchaudio.info(str(wav_file))
                duration = metadata.num_frames / metadata.sample_rate
                total_duration += duration
            except Exception as e:
                logger.warning(f"Failed to get duration for {wav_file}: {e}")
                continue

        return total_duration

    def list_datasets(self) -> List[Dict]:
        """List all created datasets with info"""
        datasets = []
        for dataset_dir in self.output_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            info_path = dataset_dir / "info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    datasets.append(info)
                except Exception as e:
                    logger.warning(f"Failed to read info for {dataset_dir.name}: {e}")
                    continue

        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)

    def cleanup(self):
        """Cleanup ASR backend resources"""
        if hasattr(self.asr_backend, "cleanup"):
            self.asr_backend.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
