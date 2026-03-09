# Repository Guidelines

## Scope

- Repository: `/Users/ia/code/projects/clis/stt`
- Public repo: `nachoal/mlx-stt`
- Primary purpose: local-first, MLX-first speech-to-text CLI for agents on macOS

## Product Boundaries

- Keep this repo focused on **transcription only**.
- Do not add summarization, general media routing, prompt-harness systems, TTS, or unrelated multimodal workflows.
- The command surface should stay small and explicit:
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

## Backend Rules

- English, short clip, speed matters: prefer `mlx-parakeet`
- English, subtitles or long audio: prefer `parakeet-mlx`
- Spanish / multilingual / unknown language: prefer `qwen3-asr-0.6b`
- Higher multilingual accuracy: prefer `qwen3-asr-1.7b`

If benchmarks change materially, update the recommendation logic, README, tests, and public repo description together.

## Portability Rules

- Do not hardcode `/Users/ia/...` paths in repo code or docs.
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
- stdlib + subprocess orchestration

Prefer keeping dependencies minimal. Add a dependency only when the gain is clear and the standard library is insufficient.

## Editing Guidance

- Preserve JSON-first output ergonomics for agent use.
- Keep stdout clean and machine-readable when `--json` is requested.
- If you add file outputs, make sure non-CLI backends can still write plain `txt/json` artifacts consistently.
- Prefer explicit code over clever abstractions.

## Validation

Run these before considering a change complete:

```bash
python3 -m py_compile stt/*.py
uv run --with pytest pytest
stt doctor --json
```

If you change recommendation logic, also run:

```bash
stt recommend /path/to/file --json
stt transcribe /path/to/file --json
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
