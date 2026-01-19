# Audio Batch Converter

A Python tool for batch transcribing audio files to text using OpenAI's Whisper model. Runs locally on your machine — no API costs, no per-minute charges.

## Features

- **Batch processing** — Transcribe entire folders of audio files
- **Resume support** — Automatically skips already-processed files
- **Multiple output formats** — Plain text, timestamped, and JSON
- **Flexible input** — Process any folder or use the default `input_files/` directory
- **Model selection** — Choose accuracy vs. speed tradeoff

## Installation

### 1. Clone the repo

```bash
git clone git@github.com:ikswodas/audio_batch_converter.git
cd audio_batch_converter
```

### 2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install openai-whisper
```

## Usage

### Basic

```bash
# Use default input_files/ folder
python audio_batch_converter.py

# Or specify a folder
python audio_batch_converter.py /path/to/audio/folder
```

Output is automatically created as `<folder_name>_transcriptions/`.

### Options

| Flag | Description |
|------|-------------|
| `-o, --output` | Custom output directory |
| `-m, --model` | Whisper model size (tiny, base, small, medium, large) |
| `-r, --recursive` | Search subdirectories for audio files |
| `--retry` | Retry previously failed files |
| `--no-skip` | Reprocess already completed files |

### Examples

```bash
# Use a larger model for better accuracy
python audio_batch_converter.py ./my_audio -m medium

# Search subdirectories
python audio_batch_converter.py ./my_audio -r

# Custom output location
python audio_batch_converter.py ~/Downloads/case_files -o ~/Documents/transcriptions

# Retry failed files
python audio_batch_converter.py ./my_audio --retry
```

## Output Structure

```
my_audio_transcriptions/
├── plain_text/
│   └── filename_transcript.txt      # Simple text transcription
├── timestamped/
│   └── filename_timestamped.txt     # Text with timestamps
├── json/
│   └── filename_data.json           # Complete metadata
├── transcription_progress.json      # Progress tracking
├── transcription_log_YYYYMMDD.log   # Processing log
└── transcription_report_YYYYMMDD.txt
```

## Model Comparison

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fastest | Basic | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Medium | Better | ~2 GB |
| medium | Slow | Very Good | ~5 GB |
| large | Slowest | Best | ~10 GB |

Default is `base` — good balance for most use cases.

## Troubleshooting

### "No module named whisper"

Make sure the virtual environment is activated:

```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### "FP16 is not supported on CPU"

This warning is normal and can be ignored. The script automatically uses FP32 on CPU.

### Script interrupted?

Just run the same command again — it picks up where it left off.

## License

[MIT](LICENSE)
