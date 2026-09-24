# Repository Guidelines

## Scope

- Public repo: `nachoal/mlx-stt`
- Primary purpose: local-first, MLX-first speech-to-text CLI for agents on macOS

## Product Boundaries

- Keep this repo focused on **transcription**: plain transcripts and speaker-attributed transcripts (`transcribe --diarize`).
- Speaker diarization is in scope only in service of transcription: cutting audio into speech turns for the ASR backends, and labeling who said what. No standalone diarization command, speaker identification or enrollment, or voice analytics.
- Do not add summarization, general media routing, prompt-harness systems, TTS, or unrelated multimodal workflows.
- The command surface should stay small and explicit (new behavior lands as flags on these, not as new commands):
  - `setup`
  - `recommend`
  - `transcribe`
  - `benchmark`
  - `doctor`

## Positioning

- Lead with **MLX-first local STT for agents on macOS**.
- Keep the CLI name `stt`.
- Keep the repo messaging clear that the tool chooses between:
  - `mlx-parakeet`
  - `parakeet-mlx`
  - `qwen3-asr-0.6b`
  - `qwen3-asr-1.7b`
- and that `--diarize` adds speaker labels (NVIDIA Nemotron-3-Diarization, local).

## Backend Rules

- English, any length, subtitles, very long audio: prefer `mlx-parakeet`
- Spanish / multilingual / unknown language, including subtitles and very long audio: prefer `qwen3-asr-0.6b`
- Higher accuracy: prefer `qwen3-asr-1.7b`
- `parakeet-mlx` stays selectable with `--backend`, but is not recommended: it drops sentences on long audio.
- `mlx-parakeet` always transcribes per speech turn, never a whole long file: long windows with pauses make Parakeet v3 drop whole sentences (English 188 s: 37% WER whole-file vs 4% per turn).
- Never route non-English audio to `mlx-parakeet` by default. mlx-audio gives Parakeet v3 no language context, so on real non-English speech it drifts into English (seven real Spanish talks: 35% WER per turn vs 7.5% for `qwen3-asr-0.6b`). A forced `--backend mlx-parakeet` with a non-English `--language` decodes each turn in 4.75 s windows (7.9%); English turns decode whole, because those windows cost English and clean read speech.
- Validate language-routing changes on real recordings, not only on read-speech corpora such as FLEURS: FLEURS showed Parakeet winning on Spanish while real talks showed the opposite.
- Qwen3-ASR decodes whole files unless `--diarize` or subtitle output needs per-turn segments; per-turn decoding measured worse on Spanish.

If benchmarks change materially, update the recommendation logic, README, tests, and public repo description together.

## Runtime Rules

- The CLI environment stays stdlib-only. Anything that needs `mlx`, `numpy`, or `mlx-audio` runs inside the runtime through `stt/runtime_worker.py` (a subprocess that prints one JSON object on its last stdout line).
- `stt/nemotron_diar_mlx.py` is a single-file MLX port of the 🤗 Transformers Nemotron-3-Diarization implementation (mlx + numpy only). Keep it single-file, keep its parity tests passing when you touch it, and never import it from CLI-side code.
- `stt/segments.py` is shared by the CLI and the worker; keep it standard-library only so the unit tests run without MLX.
- The runtime stack is pinned in `stt/constants.py` (`RUNTIME_PACKAGES`, `PARAKEET_CLI_PACKAGE`) and the diarizer by `DIARIZATION_REVISION`. Bump pins only with a before/after transcription comparison on real audio, and never point the runtime at a local fork or editable install.
- `stt setup` rebuilds the runtime from scratch (`uv venv --clear`); that is the upgrade path.

## Portability Rules

- Do not hardcode user-specific absolute checkout paths in repo code or docs.
- Prefer the repo-managed setup flow over asking users to hand-build a separate MLX runtime.
- Keep `install.sh` and `stt setup` working together.
- Use `STT_SHARED_PYTHON` for the Python runtime that has `mlx-audio`.
- Use env vars for optional benchmark fixtures:
  - `STT_SAMPLE_ENGLISH`
  - `STT_SAMPLE_ENGLISH_TEXT`
  - `STT_SAMPLE_SPANISH`
  - `STT_SAMPLE_SPANISH_TEXT`
- Never commit `.env`, local audio fixtures, caches, or machine-specific state.

## Tech Stack

- Python 3.12
- `argparse`
- `uv` for install/dev flows
- `hatchling`
- stdlib + subprocess orchestration on the CLI side; `mlx`, `mlx-audio`, `numpy` only inside the runtime

Prefer keeping dependencies minimal. Add a dependency only when the gain is clear and the standard library is insufficient.

## Editing Guidance

- Preserve JSON-first output ergonomics for agent use.
- Keep stdout clean and machine-readable when `--json` is requested.
- In-runtime backends write `txt`, `json`, `srt`, and `vtt` from `result.segments`; keep those formats consistent with and without `--diarize`.
- Prefer explicit code over clever abstractions.

## Validation

Run these before considering a change complete:

```bash
python3 -m py_compile stt/*.py
uv run --with pytest pytest
stt doctor --json
stt setup --runtime-dir /tmp/stt-runtime-test --download-models none
```

If you change recommendation logic, segmentation, the worker, or the runtime pins, also run:

```bash
stt recommend /path/to/file --json
stt transcribe /path/to/file --json
stt transcribe /path/to/multi-speaker-file --diarize --json
stt transcribe /path/to/silence.wav --json   # must return empty text
stt benchmark --suite repo-samples --json
```

Use local fixture env vars when benchmarking the sample suite.

## Release Hygiene

- Keep the public repo metadata aligned with the code:
  - README
  - `pyproject.toml`
  - GitHub repo description
- If the backend recommendation changes, update all three in the same change.
- Keep commits focused and descriptive.
