from pathlib import Path

import pytest

from tmdb_asset_miner import MinerConfig, TMDBAssetMiner


def make_miner(root: Path) -> TMDBAssetMiner:
    return TMDBAssetMiner(
        MinerConfig(output_dir=root, groups=("movie.poster", "tv.poster")),
        api_key="not-used-by-tests",
    )


def test_catalog_survives_reopen_and_rejects_duplicate_id(tmp_path: Path) -> None:
    miner = make_miner(tmp_path)
    family_id = miner._family_id("movie.poster", 671)
    asset_id = miner._asset_id("movie.poster", 671, "/poster.jpg")
    now = miner._now()

    with miner.connection:
        miner.connection.execute(
            "INSERT INTO families VALUES(?,?,?,?,?)",
            (family_id, "movie.poster", 671, "complete", now),
        )
        row = (
            asset_id,
            family_id,
            "movie.poster",
            "movie",
            "poster",
            671,
            "Harry Potter",
            "Harry Potter",
            "2001-11-16",
            "en",
            f"images/{asset_id[:2]}/{asset_id}.jpg",
            "/poster.jpg",
            1000,
            1500,
            0,
            None,
            1,
            now,
        )
        miner.connection.execute(
            "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )

    with pytest.raises(Exception):
        with miner.connection:
            miner.connection.execute(
                "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
    miner.close()

    reopened = make_miner(tmp_path)
    assert reopened._asset_count() == 1
    assert reopened._family_count_all() == 1
    reopened.close()


def test_family_id_is_stable() -> None:
    first = TMDBAssetMiner._family_id("movie.poster", 671)
    second = TMDBAssetMiner._family_id("movie.poster", 671)
    other = TMDBAssetMiner._family_id("tv.poster", 671)

    assert first == second
    assert first != other
    assert len(first) == 24


def test_local_catalog_can_open_without_tmdb_credentials(tmp_path: Path) -> None:
    miner = TMDBAssetMiner(
        MinerConfig(output_dir=tmp_path, groups=("movie.poster",)),
    )
    assert miner._asset_count() == 0
    miner.close()


def test_hf_record_embeds_image_bytes(tmp_path: Path) -> None:
    miner = make_miner(tmp_path)
    image_path = tmp_path / "images" / "00" / "asset.jpg"
    image_path.parent.mkdir(parents=True)
    payload = b"jpeg-payload"
    image_path.write_bytes(payload)

    record = miner._hf_record(
        {
            "id": "asset",
            "group_name": "movie.poster",
            "image_path": "images/00/asset.jpg",
            "all_languages_mined": 1,
        }
    )

    assert record["image"] == {"bytes": payload, "path": "asset.jpg"}
    assert record["group"] == "movie.poster"
    assert record["all_languages_mined"] is True
    assert "image_path" not in record
    miner.close()
