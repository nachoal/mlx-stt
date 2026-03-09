from pathlib import Path

from stt import config


def test_stt_home_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "mlx-stt-home"))
    assert config.stt_home() == tmp_path / "mlx-stt-home"


def test_save_and_load_config(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "mlx-stt-home"))
    saved = config.save_config({"runtime_python": "/tmp/python", "parakeet_binary": "/tmp/parakeet"})
    assert saved.exists()
    loaded = config.load_config()
    assert loaded["runtime_python"] == "/tmp/python"
    assert loaded["parakeet_binary"] == "/tmp/parakeet"


def test_resolve_shared_python_prefers_env(monkeypatch, tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.setenv("STT_SHARED_PYTHON", str(fake_python))
    assert config.resolve_shared_python() == str(fake_python)


def test_resolve_parakeet_binary_prefers_config(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_HOME", str(tmp_path / "mlx-stt-home"))
    fake_binary = tmp_path / "parakeet-mlx"
    fake_binary.write_text("")
    config.save_config({"parakeet_binary": str(fake_binary)})
    assert config.resolve_parakeet_binary() == str(fake_binary)
