import io
import json
import tarfile

from omegaconf import OmegaConf

from iimt.synthetic import pipeline
from synthesize_pairs import (
    Curve,
    _assign_split,
    _resolve_hf_repo,
    _resolve_hf_token,
    _shared_masks,
    _write_pairs,
    prepare_massive,
)


def test_expected_render_failure_is_quarantined(monkeypatch):
    pipeline._RENDER_CONTEXT = {"quality": {"quarantine_unexpected_errors": True}}

    def fail(_task):
        raise pipeline.TextDoesNotFitError("does not fit")

    monkeypatch.setattr(pipeline, "_render_family", fail)
    intervention = {
        "name": "reference",
        "intervention_type": "observational",
        "changed_factor": "none",
        "style_level": 0,
        "layout_level": 0,
        "background_level": 0,
        "index": 0,
    }
    result = pipeline._render_family_task(("family", {"en": "hello", "de": "hallo"}, "train", intervention))

    assert result.assets == []
    assert result.rejection["reason"] == "placement_failed"
    assert result.rejection["family_id"] == "family-reference"


def test_shared_layout_masks_are_language_independent():
    curve = Curve((10, 50), (35, 10), (65, 10), (90, 50))
    region, baseline = _shared_masks(curve, (100, 80), region_width=30)

    assert region.getbbox() is not None
    assert baseline.getbbox() is not None
    assert region.size == baseline.size == (100, 80)


def test_split_is_deterministic():
    config = OmegaConf.create({"split": {"train": 0.8, "validation": 0.1, "test": 0.1, "seed": 42}})
    assert _assign_split("family-1", config) == _assign_split("family-1", config)


def test_pivot_pairs_are_bidirectional():
    config = OmegaConf.create(
        {"pairing": {"mode": "pivot", "pivot_languages": ["en-US", "en"], "bidirectional": True}}
    )
    assets = [
        {
            "id": "a",
            "base_family_id": "base",
            "family_id": "f",
            "intervention_type": "observational",
            "split": "train",
            "language": "en-US",
            "text": "hello",
        },
        {
            "id": "b",
            "base_family_id": "base",
            "family_id": "f",
            "intervention_type": "observational",
            "split": "train",
            "language": "ru-RU",
            "text": "привет",
        },
        {
            "id": "c",
            "base_family_id": "base",
            "family_id": "f",
            "intervention_type": "observational",
            "split": "train",
            "language": "de-DE",
            "text": "hallo",
        },
    ]
    stream = io.StringIO()

    assert _write_pairs(stream, assets, config) == 4
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert {(row["source_language"], row["target_language"]) for row in rows} == {
        ("en-US", "ru-RU"),
        ("en-US", "de-DE"),
        ("ru-RU", "en-US"),
        ("de-DE", "en-US"),
    }


def test_prepare_massive_builds_multiway_groups(tmp_path):
    archive_path = tmp_path / "massive.tar.gz"
    output_path = tmp_path / "translations.jsonl"
    records = [
        {
            "id": "7",
            "locale": locale,
            "partition": "dev",
            "scenario": "alarm",
            "intent": "alarm_set",
            "utt": text,
        }
        for locale, text in {
            "en-US": "set alarm now",
            "ru-RU": "поставь будильник сейчас",
            "de-DE": "stelle jetzt den wecker",
        }.items()
    ]
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("data/all.jsonl")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    config = OmegaConf.create(
        {
            "prepare": {
                "url": "https://unused.invalid/file.tar.gz",
                "archive_path": str(archive_path),
                "output_path": str(output_path),
                "force": False,
                "partitions": ["train", "dev", "test"],
                "locales": ["en-US", "ru-RU", "de-DE"],
                "min_words": 2,
                "max_words": 10,
                "max_characters": 80,
                "min_languages_per_group": 3,
                "max_groups": None,
            }
        }
    )

    prepare_massive(config)
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["family_id"] == "massive-dev-7"
    assert row["split"] == "validation"
    assert set(row["translations"]) == {"en-US", "ru-RU", "de-DE"}


def test_hf_token_is_loaded_from_json(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    credentials = tmp_path / "secrets.json"
    credentials.write_text('{"hf_token": "hf_test_token"}', encoding="utf-8")
    config = OmegaConf.create(
        {"hf": {"token": None, "credentials_file": str(credentials), "token_key": "hf_token"}}
    )

    assert _resolve_hf_token(config) == "hf_test_token"


def test_hf_repo_is_loaded_from_json(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    credentials = tmp_path / "secrets.json"
    credentials.write_text('{"hf_dataset_repo": "owner/dataset"}', encoding="utf-8")
    config = OmegaConf.create(
        {"hf": {"repo": None, "credentials_file": str(credentials), "repo_key": "hf_dataset_repo"}}
    )

    assert _resolve_hf_repo(config) == "owner/dataset"
