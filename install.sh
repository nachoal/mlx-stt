#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${MLX_STT_REPO_URL:-git+https://github.com/nachoal/mlx-stt}"
DOWNLOAD_MODELS="${MLX_STT_DOWNLOAD_MODELS:-core}"
INSTALL_FFMPEG="${MLX_STT_INSTALL_FFMPEG:-1}"

log() {
  printf '[mlx-stt] %s\n' "$1"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  log "uv not found, installing it first"
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    log "uv install finished but uv is still not in PATH"
    log "Add ~/.local/bin to PATH and re-run the installer"
    exit 1
  fi
}

main() {
  if [ "$(uname -s)" != "Darwin" ]; then
    log "This installer is optimized for macOS / Apple Silicon"
  fi

  ensure_uv

  log "Installing stt CLI"
  uv tool install --force "$REPO_URL"

  setup_args=(setup --download-models "$DOWNLOAD_MODELS")
  if [ "$INSTALL_FFMPEG" = "1" ]; then
    setup_args+=(--install-ffmpeg)
  fi

  log "Bootstrapping isolated runtime"
  stt "${setup_args[@]}"

  log "Done"
  log "Try: stt doctor --json"
}

main "$@"
