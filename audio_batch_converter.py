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
        output_dir: str,
        model_size: str = "base",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.output_dir / "transcription_progress.json"
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

    def _get_relative_output_path(self, wav_file: Path) -> Path:
        """Get the output path preserving directory structure"""
        try:
            relative = wav_file.parent.relative_to(self.input_dir)
        except ValueError:
            relative = Path(".")
        return relative

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

            # Get relative path for preserving structure
            relative_path = self._get_relative_output_path(wav_file)

            # Create subdirectories for organized output
            plain_text_dir = self.output_dir / "plain_text" / relative_path
            timestamped_dir = self.output_dir / "timestamped" / relative_path
            json_dir = self.output_dir / "json" / relative_path

            plain_text_dir.mkdir(parents=True, exist_ok=True)
            timestamped_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)

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

    def process_batch(self, skip_completed: bool = True):
        """Process all WAV files in the input directory recursively"""
        batch_start_time = time.time()
        
        # Find all WAV files recursively
        wav_files = list(self.input_dir.rglob("*.wav"))
        
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
                    "path": str(wav_file),
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
                    "path": str(wav_file),
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
            filepath = info.get("path", self.input_dir / info["filename"])
            wav_file = Path(filepath)

            if not wav_file.exists():
                self.logger.warning(f"File not found: {wav_file}")
                continue

            self.logger.info(f"Retrying: {wav_file.name}")
            result = self.transcribe_file(wav_file)

            if result["success"]:
                # Remove from failed, add to completed
                del self.progress["failed"][file_hash]
                self.progress["completed"][file_hash] = {
                    "filename": wav_file.name,
                    "path": str(wav_file),
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


def process_input_files(script_dir: Path, output_base: Path, model_size: str, retry: bool, no_skip: bool):
    """Process all folders and loose files in input_files/"""
    input_base = script_dir / "input_files"
    
    # Get all top-level items in input_files
    top_level_items = list(input_base.iterdir()) if input_base.exists() else []
    
    # Separate folders and loose wav files
    folders = [item for item in top_level_items if item.is_dir()]
    loose_wavs = list(input_base.glob("*.wav"))
    
    # Process each folder as its own batch
    for folder in folders:
        output_dir = output_base / f"{folder.name}_transcribed"
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        transcriber = WhisperBatchTranscriber(
            input_dir=str(folder),
            output_dir=str(output_dir),
            model_size=model_size,
        )
        
        if retry:
            transcriber.retry_failed()
        else:
            transcriber.process_batch(skip_completed=not no_skip)
        
        transcriber.generate_report()
    
    # Process loose wav files if any
    if loose_wavs:
        output_dir = output_base / "loose_files_transcribed"
        print(f"\n{'='*60}")
        print(f"Processing loose WAV files in input_files/")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
        
        transcriber = WhisperBatchTranscriber(
            input_dir=str(input_base),
            output_dir=str(output_dir),
            model_size=model_size,
        )
        
        # Only process files directly in input_files, not in subfolders
        # We need to temporarily override to non-recursive for loose files
        wav_files = list(input_base.glob("*.wav"))
        if wav_files:
            if retry:
                transcriber.retry_failed()
            else:
                # Custom processing for just loose files
                batch_start_time = time.time()
                processed_count = 0
                skipped_count = 0
                failed_count = 0
                
                for idx, wav_file in enumerate(wav_files, 1):
                    if not no_skip and transcriber._is_already_processed(wav_file):
                        skipped_count += 1
                        transcriber.logger.info(f"[{idx}/{len(wav_files)}] Skipping: {wav_file.name}")
                        continue
                    
                    transcriber.logger.info(f"[{idx}/{len(wav_files)}] Processing: {wav_file.name}")
                    result = transcriber.transcribe_file(wav_file)
                    file_hash = transcriber._get_file_hash(wav_file)
                    
                    if result["success"]:
                        transcriber.progress["completed"][file_hash] = {
                            "filename": wav_file.name,
                            "path": str(wav_file),
                            "processed_at": datetime.now().isoformat(),
                            "duration_seconds": result["duration"],
                            "text_length": result["text_length"],
                        }
                        processed_count += 1
                    else:
                        transcriber.progress["failed"][file_hash] = {
                            "filename": wav_file.name,
                            "path": str(wav_file),
                            "failed_at": datetime.now().isoformat(),
                            "error": result["error"],
                        }
                        failed_count += 1
                    
                    transcriber._save_progress()
                
                batch_elapsed = time.time() - batch_start_time
                transcriber.logger.info("=" * 60)
                transcriber.logger.info(f"Loose files complete: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")
                transcriber.logger.info(f"Time: {batch_elapsed:.2f} seconds")
                transcriber.logger.info("=" * 60)
            
            transcriber.generate_report()


def main():
    script_dir = Path(__file__).parent
    input_files_dir = script_dir / "input_files"
    output_files_dir = script_dir / "output_files"
    
    # Ensure directories exist
    input_files_dir.mkdir(exist_ok=True)
    output_files_dir.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using Whisper"
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=None,
        help="Path to folder containing audio files (default: processes all folders in input_files/)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: output_files/<folder>_transcribed)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        help="Whisper model size (tiny, base, small, medium, large)",
        default="base"
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

    # If no input folder specified, process everything in input_files/
    if args.input_folder is None:
        process_input_files(script_dir, output_files_dir, args.model, args.retry, args.no_skip)
    else:
        # Process specific folder
        input_path = Path(args.input_folder)
        
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = output_files_dir / f"{input_path.name}_transcribed"
        
        transcriber = WhisperBatchTranscriber(
            input_dir=str(input_path),
            output_dir=str(output_path),
            model_size=args.model,
        )

        if args.retry:
            transcriber.retry_failed()
        else:
            transcriber.process_batch(skip_completed=not args.no_skip)

        transcriber.generate_report()


if __name__ == "__main__":
    main()
