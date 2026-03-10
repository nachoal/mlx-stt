# stt

`stt` is an MLX-first local speech-to-text CLI for macOS, designed for agents and automation.

It answers one practical question cleanly:

Which local transcription backend should I use for this file right now, and then how do I transcribe it?

It is intentionally focused on transcription only:

- recommend the best local backend
- run the transcript
- benchmark the local backends
- inspect the current runtime

No summarization layer. No “media router”. No prompt framework.

## Why This Exists

Local transcription on Apple Silicon is fragmented:

- `parakeet-mlx` is great for fast local English transcription and subtitle-style workflows
- direct `mlx-audio` Parakeet can be even faster on short clips
- `Qwen3-ASR` is better for Spanish and multilingual audio

Agents need a small, deterministic MLX-first CLI that tells them which one to use, then runs it.

That is what `stt` does.

## What It Uses

Models:

- [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)
- [`mlx-community/Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit)
- [`mlx-community/Qwen3-ASR-1.7B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)

Open-source projects:

- [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) for direct MLX-backed inference
- [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx) for the CLI subtitle/transcription path
- [`ffmpeg`](https://ffmpeg.org/) for media conversion and probing

## Best Local Defaults

- English, short clip, speed matters: `mlx-parakeet`
- English, subtitles or long audio: `parakeet-mlx`
- Spanish / multilingual / unknown language: `qwen3-asr-0.6b`
- Higher multilingual accuracy: `qwen3-asr-1.7b`

These defaults are based on local benchmarking logic built into the tool.

## Installation

### Fastest install

This gives you:

- the `stt` CLI
- an isolated runtime under `~/Library/Application Support/mlx-stt`
- `mlx-audio`
- `parakeet-mlx`
- pre-downloaded core models

```bash
curl -fsSL https://raw.githubusercontent.com/nachoal/mlx-stt/main/install.sh | bash
```

### Homebrew-oriented install

If you prefer to start from Homebrew-managed tools:

```bash
brew install uv ffmpeg
uv tool install git+https://github.com/nachoal/mlx-stt
stt setup --download-models core
```

### Manual install

```bash
uv tool install git+https://github.com/nachoal/mlx-stt
```

For local development:

```bash
uv tool install --force -e .
```

Then create the isolated runtime:

```bash
stt setup --download-models core
```

This creates a dedicated runtime and stores its paths in `~/Library/Application Support/mlx-stt/config.json`.

If you already have your own MLX Python environment and want to use that instead:

```bash
export STT_SHARED_PYTHON=/path/to/python-with-mlx-audio
```

`stt doctor --json` will show exactly which runtime is active.

## Commands

### Recommend

```bash
stt recommend /path/to/file.wav --language english --speed --json
```

Example output:

```json
{
  "backend": "mlx-parakeet",
  "model": "mlx-community/parakeet-tdt-0.6b-v3"
}
```

### Transcribe

```bash
stt transcribe /path/to/file.wav --language english --speed --json
```

`stt transcribe` automatically normalizes video inputs and compressed audio that benefits from ffmpeg preprocessing, including Telegram-style `.ogg`/Opus voice notes, into mono 16 kHz WAV before handing the file to the selected backend.

Force a backend:

```bash
stt transcribe /path/to/file.wav --backend qwen3-asr-0.6b --json
stt transcribe /path/to/file.wav --backend mlx-parakeet --json
stt transcribe /path/to/file.wav --backend parakeet-mlx --output-format srt --json
```

Write output files:

```bash
stt transcribe /path/to/file.wav --output-dir ./out --output-name transcript --json
```

### Benchmark

Single file:

```bash
stt benchmark /path/to/file.wav --reference-text "expected transcript" --language spanish --json
```

Fixture-based suite:

```bash
export STT_SAMPLE_ENGLISH=/path/to/english.wav
export STT_SAMPLE_ENGLISH_TEXT="Hello. This is a test."
export STT_SAMPLE_SPANISH=/path/to/spanish.wav
export STT_SAMPLE_SPANISH_TEXT="..."
stt benchmark --suite repo-samples --json
```

### Doctor

```bash
stt doctor --json
```

This reports:

- whether `ffmpeg` is installed
- whether `parakeet-mlx` is in `PATH`
- which Python runtime will be used for `mlx-audio`
- detected versions for `parakeet-mlx`, `mlx-audio`, and `transformers`

### Setup

```bash
stt setup --download-models core
```

Options:

- `--download-models none|core|all`
- `--install-ffmpeg`
- `--runtime-dir /custom/path`

## Environment

- `STT_SHARED_PYTHON`: Python executable with `mlx-audio` installed
- `STT_SAMPLE_ENGLISH`: optional benchmark fixture
- `STT_SAMPLE_ENGLISH_TEXT`: optional reference transcript
- `STT_SAMPLE_SPANISH`: optional benchmark fixture
- `STT_SAMPLE_SPANISH_TEXT`: optional reference transcript

## Tests

Run unit tests:

```bash
uv run --with pytest pytest
```

The tests cover:

- backend recommendation logic
- benchmark row shaping
- fixture-driven benchmark suite behavior

## Design Notes

This project is optimized for agent ergonomics:

- small command surface
- JSON-first output
- explicit backend recommendation
- deterministic local execution

It is inspired by the thin, shell-friendly CLI style used in several of steipete’s OSS tools, while staying Python-native because the actual local MLX/Qwen inference stack is Python-first.
