import hashlib
import json
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import whisper


class WhisperBatchTranscriber:
    def __init__(
        self,
        input_dir: str,
        output_dir: str = None,
        model_size: str = "base",
        progress_file: str = None,
    ):
        self.input_dir = Path(input_dir)
        
        # Auto-generate output dir based on input folder name
        if output_dir is None:
            self.output_dir = self.input_dir.parent / f"{self.input_dir.name}_transcriptions"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Progress file lives in output dir by default
        if progress_file is None:
            self.progress_file = self.output_dir / "transcription_progress.json"
        else:
            self.progress_file = Path(progress_file)
        
        self.model_size = model_size

        # Set up logging
        log_file = (
            self.output_dir
            / f"transcription_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # Load or initialize progress tracking
        self.progress = self._load_progress()

        # Load model once
        self.logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.logger.info("Model loaded successfully")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _load_progress(self) -> dict:
        """Load progress from JSON file or create new progress dict"""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {
            "completed": {},
            "failed": {},
            "stats": {"total_processed": 0, "total_failed": 0, "total_time_seconds": 0},
        }

    def _save_progress(self):
        """Save progress to JSON file"""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def _get_file_hash(self, filepath: Path) -> str:
        """Generate MD5 hash of file for tracking"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_already_processed(self, wav_file: Path) -> bool:
        """Check if file has already been successfully processed"""
        file_hash = self._get_file_hash(wav_file)
        return file_hash in self.progress["completed"]

    def transcribe_file(self, wav_file: Path) -> Optional[dict]:
        """Transcribe a single WAV file"""
        try:
            start_time = time.time()
            self.logger.info(f"Processing: {wav_file.name}")

            # Transcribe
            result = self.model.transcribe(
                str(wav_file),
                verbose=False,
                language="en",
            )

            # Create subdirectories for organized output
            plain_text_dir = self.output_dir / "plain_text"
            timestamped_dir = self.output_dir / "timestamped"
            json_dir = self.output_dir / "json"

            plain_text_dir.mkdir(exist_ok=True)
            timestamped_dir.mkdir(exist_ok=True)
            json_dir.mkdir(exist_ok=True)

            # Save plain text transcription
            txt_file = plain_text_dir / f"{wav_file.stem}_transcript.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(result["text"])

            # Save timestamped version
            timestamped_file = timestamped_dir / f"{wav_file.stem}_timestamped.txt"
            with open(timestamped_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    timestamp = f"[{self._format_timestamp(segment['start'])} -> {self._format_timestamp(segment['end'])}]"
                    f.write(f"{timestamp} {segment['text']}\n")

            # Save full result with timestamps
            json_file = json_dir / f"{wav_file.stem}_data.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            elapsed = time.time() - start_time

            return {
                "success": True,
                "txt_file": str(txt_file),
                "json_file": str(json_file),
                "duration": elapsed,
                "text_length": len(result["text"]),
            }

        except Exception as e:
            self.logger.error(f"Failed to process {wav_file.name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_batch(self, skip_completed: bool = True, recursive: bool = False):
        """Process all WAV files in the input directory"""
        batch_start_time = time.time()
        
        # Find all WAV files
        if recursive:
            wav_files = list(self.input_dir.rglob("*.wav"))
        else:
            wav_files = list(self.input_dir.glob("*.wav"))
        
        total_files = len(wav_files)

        if total_files == 0:
            self.logger.warning(f"No WAV files found in {self.input_dir}")
            return

        self.logger.info(f"Found {total_files} WAV files to process")

        processed_count = 0
        skipped_count = 0
        failed_count = 0

        for idx, wav_file in enumerate(wav_files, 1):
            # Check if already processed
            if skip_completed and self._is_already_processed(wav_file):
                skipped_count += 1
                self.logger.info(
                    f"[{idx}/{total_files}] Skipping (already processed): {wav_file.name}"
                )
                continue

            # Process file
            self.logger.info(f"[{idx}/{total_files}] Processing: {wav_file.name}")
            result = self.transcribe_file(wav_file)

            # Update progress
            file_hash = self._get_file_hash(wav_file)

            if result["success"]:
                self.progress["completed"][file_hash] = {
                    "filename": wav_file.name,
                    "processed_at": datetime.now().isoformat(),
                    "duration_seconds": result["duration"],
                    "text_length": result["text_length"],
                }
                processed_count += 1
                self.progress["stats"]["total_processed"] += 1
                self.progress["stats"]["total_time_seconds"] += result["duration"]
            else:
                self.progress["failed"][file_hash] = {
                    "filename": wav_file.name,
                    "failed_at": datetime.now().isoformat(),
                    "error": result["error"],
                }
                failed_count += 1
                self.progress["stats"]["total_failed"] += 1

            # Save progress after each file
            self._save_progress()

        # Calculate actual batch elapsed time
        batch_elapsed = time.time() - batch_start_time

        # Final summary
        self.logger.info("=" * 60)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info(f"Total files found: {total_files}")
        self.logger.info(f"Newly processed: {processed_count}")
        self.logger.info(f"Skipped (already done): {skipped_count}")
        self.logger.info(f"Failed: {failed_count}")
        self.logger.info(f"This batch took: {batch_elapsed:.2f} seconds ({batch_elapsed/60:.1f} minutes)")
        self.logger.info("=" * 60)

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def retry_failed(self):
        """Retry all previously failed transcriptions"""
        failed_files = self.progress["failed"].copy()

        if not failed_files:
            self.logger.info("No failed files to retry")
            return

        self.logger.info(f"Retrying {len(failed_files)} failed files")

        for file_hash, info in failed_files.items():
            filename = info["filename"]
            wav_file = self.input_dir / filename

            if not wav_file.exists():
                self.logger.warning(f"File not found: {filename}")
                continue

            self.logger.info(f"Retrying: {filename}")
            result = self.transcribe_file(wav_file)

            if result["success"]:
                # Remove from failed, add to completed
                del self.progress["failed"][file_hash]
                self.progress["completed"][file_hash] = {
                    "filename": filename,
                    "processed_at": datetime.now().isoformat(),
                    "duration_seconds": result["duration"],
                    "text_length": result["text_length"],
                }
                self.progress["stats"]["total_processed"] += 1
                self.progress["stats"]["total_time_seconds"] += result["duration"]

            self._save_progress()

    def generate_report(self):
        """Generate a summary report"""
        report_file = (
            self.output_dir
            / f"transcription_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        with open(report_file, "w") as f:
            f.write("TRANSCRIPTION BATCH REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model used: {self.model_size}\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(
                f"Total files processed: {self.progress['stats']['total_processed']}\n"
            )
            f.write(f"Total files failed: {self.progress['stats']['total_failed']}\n")
            f.write(
                f"Total processing time: {self.progress['stats']['total_time_seconds']:.2f} seconds\n"
            )
            f.write(
                f"Average time per file: {self.progress['stats']['total_time_seconds'] / max(self.progress['stats']['total_processed'], 1):.2f} seconds\n\n"
            )

            if self.progress["failed"]:
                f.write("FAILED FILES:\n")
                f.write("-" * 60 + "\n")
                for file_hash, info in self.progress["failed"].items():
                    f.write(f"{info['filename']}: {info['error']}\n")

        self.logger.info(f"Report generated: {report_file}")


def main():
    # Ensure input directory exists
    script_dir = Path(__file__).parent
    default_input = script_dir / "input_files"
    default_input.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description="Batch transcribe WAV files using Whisper"
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=str(default_input),
        help="Path to folder containing WAV files (default: ./input_files)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: <input_folder>_transcriptions)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        help="Whisper model size (tiny, base, small, medium, large)",
        default="base"
    )
    parser.add_argument(
        "-r", "--recursive",
        help="Search for WAV files recursively in subdirectories",
        action="store_true"
    )
    parser.add_argument(
        "--retry",
        help="Retry previously failed files",
        action="store_true"
    )
    parser.add_argument(
        "--no-skip",
        help="Reprocess already completed files",
        action="store_true"
    )

    args = parser.parse_args()

    transcriber = WhisperBatchTranscriber(
        input_dir=args.input_folder,
        output_dir=args.output,
        model_size=args.model,
    )

    if args.retry:
        transcriber.retry_failed()
    else:
        transcriber.process_batch(
            skip_completed=not args.no_skip,
            recursive=args.recursive
        )

    transcriber.generate_report()


if __name__ == "__main__":
    main()
