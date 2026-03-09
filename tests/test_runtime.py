from pathlib import Path

from stt.runtime import bootstrap_runtime


def test_bootstrap_runtime_writes_config(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    home_dir = tmp_path / "home"
    commands: list[list[str]] = []
    warmed: list[str] = []

    monkeypatch.setenv("STT_HOME", str(home_dir))
    monkeypatch.setattr("stt.runtime.which", lambda name: "/opt/homebrew/bin/uv" if name == "uv" else "/opt/homebrew/bin/ffmpeg")

    def fake_run(cmd, check=True, live=False):
        commands.append(cmd)
        if cmd[:2] == ["uv", "venv"]:
            (runtime_dir / "bin").mkdir(parents=True, exist_ok=True)
            (runtime_dir / "bin" / "python").write_text("")
            (runtime_dir / "bin" / "parakeet-mlx").write_text("")
        return None

    monkeypatch.setattr("stt.runtime.run_command", fake_run)
    monkeypatch.setattr("stt.runtime._warm_model", lambda runtime_python, model_id: warmed.append(model_id))

    result = bootstrap_runtime(runtime_dir=runtime_dir, download_models="core", install_ffmpeg=False)

    assert Path(result.runtime_python).exists()
    assert Path(result.parakeet_binary).exists()
    assert warmed
    assert any(cmd[:2] == ["uv", "venv"] for cmd in commands)
    assert any(cmd[:3] == ["uv", "pip", "install"] for cmd in commands)
    assert Path(result.config_path).exists()
