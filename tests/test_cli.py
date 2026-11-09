from pathlib import Path

from run_miner import load_env


def test_load_env_does_not_override_existing_value(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ONE=from-file\nTWO='quoted value'\n", encoding="utf-8")
    monkeypatch.setenv("ONE", "from-process")

    load_env(env_file)

    assert __import__("os").environ["ONE"] == "from-process"
    assert __import__("os").environ["TWO"] == "quoted value"


def test_load_missing_env_is_allowed(tmp_path: Path) -> None:
    load_env(tmp_path / "missing.env")
