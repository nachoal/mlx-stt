# stt

`stt` is an MLX-first local speech-to-text CLI for macOS, designed for agents and automation.

It answers one practical question cleanly:

Which local transcription backend should I use for this file right now, and then how do I transcribe it?

It is intentionally focused on transcription:

- recommend the best local backend
- run the transcript, optionally labeled by speaker (`--diarize`)
- benchmark the local backends
- inspect the current runtime

No summarization layer. No “media router”. No prompt framework.

## Why This Exists

Local transcription on Apple Silicon is fragmented, and each backend has a failure mode:

- Parakeet v3 is fast and accurate, but it silently drops whole sentences when it decodes long windows that contain pauses
- `Qwen3-ASR` is strong on Spanish and multilingual audio, but slower
- code-switched conversations (one speaker in Spanish, the next in English) lose one side in whole-file decoding

Agents need a small, deterministic MLX-first CLI that tells them which one to use, then runs it without those traps.

That is what `stt` does.

## What It Uses

Models:

- [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)
- [`mlx-community/Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit)
- [`mlx-community/Qwen3-ASR-1.7B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)
- [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) for speaker turns (OpenMDW-1.1 weights, 100M params)

Open-source projects:

- [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) for direct MLX-backed inference
- [`parakeet-mlx`](https://github.com/senstella/parakeet-mlx) for the optional external CLI backend
- [`ffmpeg`](https://ffmpeg.org/) for media conversion and probing

The diarizer runs through `stt/nemotron_diar_mlx.py`, a small MLX port of the 🤗 Transformers
implementation (mlx + numpy only) that loads NVIDIA's checkpoint directly. It diarizes an hour of
audio in about 3.5 s on an M3 Max.

## Best Local Defaults

- English, any length, subtitles, very long audio: `mlx-parakeet` (transcribed per speech turn)
- Spanish / multilingual / unknown language, including subtitles and long audio: `qwen3-asr-0.6b`
- Higher accuracy: `qwen3-asr-1.7b`
- `parakeet-mlx` stays available with `--backend parakeet-mlx`, but is no longer recommended: it drops sentences on long audio

These defaults are based on local benchmarking logic built into the tool.

## Speech-turn segmentation

`mlx-parakeet` never decodes a whole long file. `stt` first runs Nemotron-3-Diarization, cuts the
audio into speaker-pure speech segments (same-speaker pauses under 1 s merge, up to 20 s per segment,
cuts in the middle of silences so no speech is dropped), and transcribes each segment. With a
non-English `--language`, each segment is further decoded in 4.75 s windows: mlx-audio gives Parakeet v3
no language context, and on real non-English speech it drifts into English otherwise. Qwen3-ASR keeps
whole-file decoding unless `--diarize` or subtitle output needs per-turn segments.

Measured on a Spanish/English corpus (FLEURS clips, 188 s concatenations, a 35 s seam case, and
two-speaker code-switch clips) with `stt`'s own WER, before and after this change:

| Case | Backend | Before | After |
|---|---|---|---|
| English, 188 s | `mlx-parakeet` | 37.05% | 4.32% |
| English, 188 s | `parakeet-mlx` | 23.64% | 23.64% (unchanged, not recommended) |
| Spanish, 188 s | `mlx-parakeet` | 8.98% | 2.51% |
| Spanish, 35 s seam | `mlx-parakeet` | 12.82% | 2.56% |
| Code-switch, 2 clips | `mlx-parakeet` | 43.75% | 6.25% |
| Short FLEURS clips, 10 | `mlx-parakeet` | 4.10% | 3.80% |
| All cases | `qwen3-asr-0.6b` | unchanged or slightly better (runtime upgrade) | |

Read-speech corpora flatter Parakeet on Spanish. On seven real Spanish talks (110 min; speakers from
Spain, Mexico, and Argentina; one with human captions, six scored against YouTube auto-captions):

| Backend | WER | Speed |
|---|---|---|
| `mlx-parakeet`, speech turns only | 35.44% | 93x |
| `mlx-parakeet`, speech turns + 4.75 s windows (`--language spanish`) | 7.93% | 61x |
| `qwen3-asr-0.6b` (the Spanish default) | 7.51% | 15x |
| `qwen3-asr-1.7b` | 6.22% | 10x |

That is why Spanish stays on Qwen3-ASR, and why a forced `--backend mlx-parakeet` narrows its windows for
non-English hints. The same windows cost English and clean read speech (English FLEURS 4.3% → 21.4%), so
English segments decode whole.

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
stt transcribe /path/to/file.wav --backend parakeet-mlx --json
```

Label speakers:

```bash
stt transcribe meeting.m4a --diarize
```

```text
speaker_0: Are we we're not allowed to do the lights so people can see that a bit better?
speaker_1: Yeah.
speaker_0: Okay, that's fine. Am I supposed to be standing up there?
speaker_2: So we've got both of these clipped on. Is she gonna answer me?
```

With `--json`, `result.segments` holds timestamped segments (sentences for `mlx-parakeet`, speech
turns for Qwen3-ASR) and, with `--diarize`, a `speaker` per segment plus `result.speakers`, the count.
Speakers are numbered in order of first appearance; up to 8 per recording. `--diarize` works with
`mlx-parakeet`, `qwen3-asr-0.6b` and `qwen3-asr-1.7b`.

```json
{
  "result": {
    "backend": "mlx-parakeet",
    "text": "Are we we're not allowed to do the lights so people can see that a bit better? Yeah. ...",
    "speakers": 4,
    "segments": [
      {"start": 10.74, "end": 14.42, "text": "Are we we're not allowed to do the lights so people can see that a bit better?", "speaker": "speaker_0"},
      {"start": 17.67, "end": 18.31, "text": "Yeah.", "speaker": "speaker_1"}
    ],
    "timings": {"diarization_seconds": 1.09, "asr_load_seconds": 0.86, "asr_seconds": 13.38, "planned_segments": 293}
  }
}
```

Write output files (`txt`, `json`, `srt`, `vtt`, or `all`; subtitle cues carry `speaker_N:` with `--diarize`):

```bash
stt transcribe /path/to/file.wav --output-dir ./out --output-name transcript --output-format all --diarize --json
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
- detected versions for `mlx`, `parakeet-mlx`, `mlx-audio`, and `transformers`
- the diarization model

### Setup

```bash
stt setup --download-models core
```

Options:

- `--download-models none|core|all` (`core` also caches the diarizer)
- `--install-ffmpeg`
- `--runtime-dir /custom/path`

Re-running `setup` rebuilds the runtime from scratch on the pinned stack (`mlx==0.32.2`,
`mlx-audio==0.5.5`, `transformers==5.17.0`, `parakeet-mlx==0.5.2`, Python 3.12). That is how
upgrades land.

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
- speech-turn segmentation and transcript formats (`txt`, `srt`, `vtt`, speaker lines)
- the runtime worker contract and `--diarize` handling

## Design Notes

This project is optimized for agent ergonomics:

- small command surface
- JSON-first output
- explicit backend recommendation
- deterministic local execution

It is inspired by the thin, shell-friendly CLI style used in several of steipete’s OSS tools, while staying Python-native because the actual local MLX/Qwen inference stack is Python-first.
