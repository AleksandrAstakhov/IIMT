import io
import json

from omegaconf import OmegaConf

from synthesize_pairs import Curve, _assign_split, _shared_masks, _write_pairs


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
    config = OmegaConf.create({
        "pairing": {"mode": "pivot", "pivot_languages": ["en-US", "en"], "bidirectional": True}
    })
    assets = [
        {"id": "a", "family_id": "f", "split": "train", "language": "en-US", "text": "hello"},
        {"id": "b", "family_id": "f", "split": "train", "language": "ru-RU", "text": "привет"},
        {"id": "c", "family_id": "f", "split": "train", "language": "de-DE", "text": "hallo"},
    ]
    stream = io.StringIO()

    assert _write_pairs(stream, assets, config) == 4
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert {(row["source_language"], row["target_language"]) for row in rows} == {
        ("en-US", "ru-RU"), ("en-US", "de-DE"), ("ru-RU", "en-US"), ("de-DE", "en-US")
    }
