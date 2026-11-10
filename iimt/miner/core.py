#!/usr/bin/env python3
"""Resumable TMDB multilingual artwork miner backed by SQLite."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imagehash
import polars as pl
import requests
from PIL import Image


@dataclass(frozen=True)
class MinerConfig:
    output_dir: Path
    groups: tuple[str, ...] | str = ("movie.poster",)
    source_language: str = "en"
    target_language: str = "ru"
    max_candidates_per_language: int = 5
    max_phash_distance: int = 25
    requests_per_second: float = 5.0
    image_size: str = "w780"
    candidate_image_size: str = "w342"
    min_vote_count: int = 20
    max_discover_page: int = 500
    strict_target: bool = False
    telegram_bot_token: str | None = None
    telegram_chat_id: str | int | None = None
    telegram_progress_every: int = 100


class TMDBAssetMiner:
    HF_EXPORT_SCHEMA_VERSION = "embedded-image-bytes-v2"
    API_URL = "https://api.themoviedb.org/3"
    IMAGE_URL = "https://image.tmdb.org/t/p"
    GROUPS = {
        "movie.poster": (
            "/discover/movie",
            "/movie/{id}/images",
            "posters",
            "movie",
            "poster",
            "title",
            "original_title",
            "release_date",
        ),
        "movie.logo": (
            "/discover/movie",
            "/movie/{id}/images",
            "logos",
            "movie",
            "logo",
            "title",
            "original_title",
            "release_date",
        ),
        "movie.backdrop": (
            "/discover/movie",
            "/movie/{id}/images",
            "backdrops",
            "movie",
            "backdrop",
            "title",
            "original_title",
            "release_date",
        ),
        "tv.poster": (
            "/discover/tv",
            "/tv/{id}/images",
            "posters",
            "tv",
            "poster",
            "name",
            "original_name",
            "first_air_date",
        ),
        "tv.logo": (
            "/discover/tv",
            "/tv/{id}/images",
            "logos",
            "tv",
            "logo",
            "name",
            "original_name",
            "first_air_date",
        ),
        "tv.backdrop": (
            "/discover/tv",
            "/tv/{id}/images",
            "backdrops",
            "tv",
            "backdrop",
            "name",
            "original_name",
            "first_air_date",
        ),
    }

    def __init__(self, config: MinerConfig, *, api_key: str | None = None, bearer_token: str | None = None):
        if config.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.config, self.api_key, self.bearer_token = config, api_key, bearer_token
        self.root = Path(config.output_dir)
        self.images_dir, self.staging_dir = self.root / "images", self.root / "staging"
        self.export_dir, self.database_path = self.root / "export", self.root / "catalog.sqlite"
        for directory in (self.root, self.images_dir, self.staging_dir, self.export_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.groups = self._resolve_groups(config.groups)
        self.telegram_bot_token = config.telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = config.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.session, self._last_request = requests.Session(), 0.0
        self.connection = sqlite3.connect(self.database_path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()
        self._cleanup_staging()
        self._migrate_legacy_parquet()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @classmethod
    def available_groups(cls) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "group": name,
                    "media_type": spec[3],
                    "asset_type": spec[4],
                    "discover_endpoint": spec[0],
                    "images_endpoint": spec[1],
                }
                for name, spec in cls.GROUPS.items()
            ]
        ).sort("group")

    def available_languages(self) -> pl.DataFrame:
        return (
            pl.DataFrame(self._api_get("/configuration/languages"))
            .select("iso_639_1", "english_name", "name")
            .sort("english_name")
        )

    def mine(self, target_families: int) -> pl.DataFrame:
        if target_families <= 0:
            raise ValueError("target_families must be positive")
        before = self._asset_count()
        self._notify(
            f"🚀 TMDB mining started\nGroups: {', '.join(self.groups)}\n"
            f"Languages: {self.config.source_language} -> {self.config.target_language}\n"
            f"Target: {target_families} families per group\nExisting assets: {before}"
        )
        try:
            for group in self.groups:
                self._mine_group(group, target_families)
        except KeyboardInterrupt:
            self._notify(f"⏹️ TMDB mining interrupted\nAssets safely stored: {self._asset_count()}")
            raise
        except BaseException as error:
            self._notify(
                f"❌ TMDB mining failed\nError: {type(error).__name__}: {error}\n"
                f"Assets safely stored: {self._asset_count()}"
            )
            raise
        after = self._asset_count()
        self._notify(
            f"✅ TMDB mining completed\nNew assets: {after - before}\nTotal assets: {after}\n"
            f"Total families: {self._family_count_all()}"
        )
        return self.status()

    def split(self, *, train=0.8, validation=0.1, test=0.1, seed=42) -> pl.DataFrame:
        if any(x < 0 for x in (train, validation, test)) or abs(train + validation + test - 1) > 1e-9:
            raise ValueError("train + validation + test must equal 1")
        updates = []
        for (family_id,) in self.connection.execute("SELECT family_id FROM families WHERE status='complete'"):
            value = (
                int.from_bytes(hashlib.sha256(f"{seed}\x1f{family_id}".encode()).digest()[:8], "big") / 2**64
            )
            name = "train" if value < train else "validation" if value < train + validation else "test"
            updates.append((name, family_id))
            if len(updates) == 10_000:
                self.connection.executemany("UPDATE assets SET split=? WHERE family_id=?", updates)
                self.connection.commit()
                updates.clear()
        if updates:
            self.connection.executemany("UPDATE assets SET split=? WHERE family_id=?", updates)
            self.connection.commit()
        return self._read_sql(
            "SELECT split, COUNT(*) assets, COUNT(DISTINCT family_id) families "
            "FROM assets GROUP BY split ORDER BY split"
        )

    def pairs(self, source_language: str, target_language: str, *, split=None, limit=None):
        clauses, params = ["a.language=?", "b.language=?"], [source_language, target_language]
        if split is not None:
            clauses.append("a.split=?")
            params.append(split)
        suffix = f" LIMIT {int(limit)}" if limit is not None else ""
        return self._read_sql(
            f"""SELECT a.family_id, a.group_name AS "group", a.media_type,
            a.asset_type, a.media_id, a.title, a.original_title, a.release_date, a.split,
            a.id source_id, a.language source_language, a.image_path source_image,
            a.tmdb_path source_tmdb_path, b.id target_id, b.language target_language,
            b.image_path target_image, b.tmdb_path target_tmdb_path
            FROM assets a JOIN assets b ON a.family_id=b.family_id
            WHERE {" AND ".join(clauses)} ORDER BY a.family_id{suffix}""",
            params,
        )

    def query(self, sql: str, parameters=()) -> pl.DataFrame:
        if not sql.lstrip().lower().startswith(("select", "with", "pragma")):
            raise ValueError("query() accepts read-only statements")
        return self._read_sql(sql, parameters)

    def status(self):
        return self._read_sql(
            'SELECT group_name AS "group", COUNT(*) assets, '
            "COUNT(DISTINCT family_id) families, COUNT(DISTINCT language) languages "
            "FROM assets GROUP BY group_name ORDER BY group_name"
        )

    def mining_status(self):
        return self._read_sql("SELECT * FROM mining_state ORDER BY group_name")

    def mined_language_coverage(self):
        return self._read_sql(
            "SELECT language, COUNT(*) assets, COUNT(DISTINCT family_id) families "
            "FROM assets GROUP BY language ORDER BY families DESC"
        )

    def notify(self, message: str) -> bool:
        """Send an optional best-effort Telegram notification."""
        return self._notify(message)

    def push(self, repo_id: str, *, token=None, private=True, export_id="default", rows_per_shard=5_000):
        """Create one temporary HF Parquet shard at a time and upload resumably."""
        try:
            from datasets import Dataset
            from datasets import Image as HFImage
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("pip install datasets huggingface_hub") from error
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        if self.connection.execute("SELECT COUNT(*) FROM assets WHERE split IS NULL").fetchone()[0]:
            self.split()
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        for split in ("train", "validation", "test"):
            shard, last_id = 0, ""
            while True:
                rows = self.connection.execute(
                    "SELECT * FROM assets WHERE split=? AND id>? ORDER BY id LIMIT ?",
                    (split, last_id, rows_per_shard),
                ).fetchall()
                if not rows:
                    break
                records = [self._hf_record(dict(row)) for row in rows]
                checksum_payload = "\n".join(
                    [self.HF_EXPORT_SCHEMA_VERSION, *(record["id"] for record in records)]
                )
                checksum = hashlib.sha256(checksum_payload.encode()).hexdigest()
                hf_path = f"data/{split}-{shard:06d}.parquet"
                done = self.connection.execute(
                    "SELECT checksum FROM exports WHERE export_id=? AND split=? "
                    "AND shard_number=? AND status='uploaded'",
                    (export_id, split, shard),
                ).fetchone()
                if not done or done["checksum"] != checksum:
                    temporary = self.export_dir / f"{export_id}-{split}-{shard:06d}.parquet"
                    Dataset.from_list(records).cast_column("image", HFImage()).to_parquet(str(temporary))
                    api.upload_file(
                        path_or_fileobj=str(temporary),
                        path_in_repo=hf_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Upload {split} shard {shard}",
                    )
                    with self.connection:
                        self.connection.execute(
                            """INSERT INTO exports VALUES(?,?,?,?,?,?,?)
                          ON CONFLICT(export_id,split,shard_number) DO UPDATE SET status=excluded.status,
                          checksum=excluded.checksum,hf_path=excluded.hf_path,updated_at_utc=excluded.updated_at_utc""",
                            (export_id, split, shard, "uploaded", checksum, hf_path, self._now()),
                        )
                    temporary.unlink(missing_ok=True)
                last_id, shard = rows[-1]["id"], shard + 1

    def _mine_group(self, group, target):
        spec = self.GROUPS[group]
        state = self.connection.execute(
            "SELECT next_page FROM mining_state WHERE group_name=? AND "
            "source_language=? AND target_language=?",
            (group, self.config.source_language, self.config.target_language),
        ).fetchone()
        page = int(state["next_page"]) if state else 1
        count, reason = self._completed_family_count(group), "max_discover_page"
        initial = count
        print(
            f"{group} {self.config.source_language}<->{self.config.target_language}: {count}/{target}, page={page}"
        )
        while count < target and page <= self.config.max_discover_page:
            payload = self._api_get(
                spec[0],
                {
                    "page": page,
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "vote_count.gte": self.config.min_vote_count,
                },
            )
            entities = payload.get("results", [])
            if not entities:
                reason = "api_returned_no_results"
                break
            for entity in entities:
                if count >= target:
                    break
                media_id = int(entity["id"])
                if self._has_complete_family(group, media_id):
                    continue
                try:
                    match = self._best_match(group, media_id)
                    if match is not None and self._save_family(group, spec, entity, match):
                        count += 1
                        print(f"[{count}/{target}] {entity.get(spec[5])}")
                        n = self.config.telegram_progress_every
                        if n > 0 and count % n == 0:
                            self._notify(
                                f"📊 TMDB progress\nGroup: {group}\nFamilies: {count}/{target}\n"
                                f"Page: {page}\nAssets: {self._asset_count()}"
                            )
                except KeyboardInterrupt:
                    self._write_state(group, page, count, target, False, None)
                    raise
                except Exception as error:
                    print(f"Skip {group}:{media_id}: {type(error).__name__}: {error}")
            page += 1
            self._write_state(group, page, count, target, False, None)
        if count < target:
            self._write_state(group, page, count, target, True, reason)
            message = f"{group}: collected {count}/{target}; {reason}"
            self._notify(f"⚠️ Group exhausted\n{message}")
            if self.config.strict_target:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            self._notify(
                f"✅ Group completed\nGroup: {group}\nFamilies: {count}/{target}\nNew: {count - initial}"
            )

    def _best_match(self, group: str, media_id: int) -> dict[str, Any] | None:
        spec = self.GROUPS[group]
        assets = self._api_get(spec[1].format(id=media_id)).get(spec[2], [])

        def top(language):
            rows = [x for x in assets if x.get("iso_639_1") == language]
            return sorted(
                rows, key=lambda x: (x.get("vote_count", 0), x.get("vote_average", 0)), reverse=True
            )[: self.config.max_candidates_per_language]

        def features(rows):
            output = []
            for row in rows:
                try:
                    image = self._download(row["file_path"], self.config.candidate_image_size)
                    w, h = image.size
                    crop = image.crop((int(0.08 * w), int(0.06 * h), int(0.92 * w), int(0.78 * h)))
                    output.append((row, imagehash.phash(crop, hash_size=16)))
                except requests.RequestException:
                    pass
            return output

        left = features(top(self.config.source_language))
        if not left:
            return None
        if self.config.target_language != "all":
            choices = [
                {"distance": int(ha - hb), "left": a, "right": b}
                for a, ha in left
                for b, hb in features(top(self.config.target_language))
            ]
            if not choices:
                return None
            best = min(choices, key=lambda x: x["distance"])
            return best if best["distance"] <= self.config.max_phash_distance else None
        grouped: dict[str, list[dict[str, Any]]] = {}
        for asset in assets:
            language = asset.get("iso_639_1")
            if language and language != self.config.source_language:
                grouped.setdefault(language, []).append(asset)
        targets = {
            lang: features(
                sorted(rows, key=lambda x: (x.get("vote_count", 0), x.get("vote_average", 0)), reverse=True)[
                    : self.config.max_candidates_per_language
                ]
            )
            for lang, rows in grouped.items()
        }
        choices = []
        for anchor, anchor_hash in left:
            matches = []
            for lang, candidates in targets.items():
                if not candidates:
                    continue
                asset, target_hash = min(candidates, key=lambda item: int(anchor_hash - item[1]))
                distance = int(anchor_hash - target_hash)
                if distance <= self.config.max_phash_distance:
                    matches.append({"language": lang, "asset": asset, "distance": distance})
            if matches:
                choices.append(
                    {
                        "anchor": anchor,
                        "targets": matches,
                        "mean": sum(x["distance"] for x in matches) / len(matches),
                    }
                )
        return min(choices, key=lambda x: (-len(x["targets"]), x["mean"])) if choices else None

    def _save_family(self, group, spec, entity, match):
        media_id = int(entity["id"])
        if self.config.target_language == "all":
            candidates = [(self.config.source_language, match["anchor"], 0)] + [
                (x["language"], x["asset"], int(x["distance"])) for x in match["targets"]
            ]
        else:
            candidates = [
                (self.config.source_language, match["left"], int(match["distance"])),
                (self.config.target_language, match["right"], int(match["distance"])),
            ]
        family_id = self._family_id(group, media_id)
        stage = self.staging_dir / family_id
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        prepared = []
        try:
            for language, asset, distance in candidates:
                asset_id, final = self._asset_id(group, media_id, asset["file_path"]), None
                final = self._image_path(asset_id)
                temporary = stage / f"{asset_id}.jpg"
                if not final.exists():
                    self._download(asset["file_path"], self.config.image_size).save(
                        temporary, "JPEG", quality=95
                    )
                    with Image.open(temporary) as check:
                        check.verify()
                prepared.append((language, asset, distance, asset_id, temporary, final))
            with self.connection:
                self.connection.execute(
                    """INSERT INTO families VALUES(?,?,?,?,?)
                  ON CONFLICT(family_id) DO UPDATE SET status=excluded.status,updated_at_utc=excluded.updated_at_utc""",
                    (family_id, group, media_id, "pending", self._now()),
                )
                for language, asset, distance, asset_id, temporary, final in prepared:
                    final.parent.mkdir(parents=True, exist_ok=True)
                    if temporary.exists() and not final.exists():
                        os.replace(temporary, final)
                    self.connection.execute(
                        "INSERT OR IGNORE INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            asset_id,
                            family_id,
                            group,
                            spec[3],
                            spec[4],
                            media_id,
                            entity.get(spec[5]),
                            entity.get(spec[6]),
                            entity.get(spec[7]),
                            language,
                            str(final.relative_to(self.root)),
                            asset["file_path"],
                            int(asset.get("width") or 0),
                            int(asset.get("height") or 0),
                            distance,
                            None,
                            int(self.config.target_language == "all"),
                            self._now(),
                        ),
                    )
                self.connection.execute(
                    "UPDATE families SET status='complete',updated_at_utc=? WHERE family_id=?",
                    (self._now(), family_id),
                )
            return True
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _create_schema(self):
        self.connection.executescript("""PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS families(family_id TEXT PRIMARY KEY,group_name TEXT NOT NULL,
          media_id INTEGER NOT NULL,status TEXT NOT NULL,updated_at_utc TEXT NOT NULL,
          UNIQUE(group_name,media_id));
        CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,family_id TEXT NOT NULL REFERENCES families(family_id),
          group_name TEXT NOT NULL,media_type TEXT NOT NULL,asset_type TEXT NOT NULL,media_id INTEGER NOT NULL,
          title TEXT,original_title TEXT,release_date TEXT,language TEXT NOT NULL,image_path TEXT NOT NULL,
          tmdb_path TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,match_phash_distance INTEGER,
          split TEXT,all_languages_mined INTEGER NOT NULL,collected_at_utc TEXT NOT NULL,
          UNIQUE(group_name,media_id,tmdb_path));
        CREATE INDEX IF NOT EXISTS assets_family_idx ON assets(family_id);
        CREATE INDEX IF NOT EXISTS assets_split_id_idx ON assets(split,id);
        CREATE INDEX IF NOT EXISTS assets_language_idx ON assets(language);
        CREATE INDEX IF NOT EXISTS assets_group_media_idx ON assets(group_name,media_id);
        CREATE TABLE IF NOT EXISTS mining_state(group_name TEXT,source_language TEXT,target_language TEXT,
          next_page INTEGER,achieved_families INTEGER,target_families INTEGER,exhausted INTEGER,
          exhaustion_reason TEXT,updated_at_utc TEXT,PRIMARY KEY(group_name,source_language,target_language));
        CREATE TABLE IF NOT EXISTS exports(export_id TEXT,split TEXT,shard_number INTEGER,status TEXT,
          checksum TEXT,hf_path TEXT,updated_at_utc TEXT,PRIMARY KEY(export_id,split,shard_number));""")

    def _migrate_legacy_parquet(self):
        legacy, marker = self.root / "assets.parquet", self.root / ".assets_parquet_migrated"
        if not legacy.exists() or marker.exists() or self._asset_count():
            return
        print(f"Migrating legacy manifest: {legacy}")
        with self.connection:
            for row in pl.read_parquet(legacy).iter_rows(named=True):
                group, image = (
                    row.get("group") or row.get("group_name"),
                    row.get("image") or row.get("image_path"),
                )
                self.connection.execute(
                    "INSERT OR IGNORE INTO families VALUES(?,?,?,?,?)",
                    (row["family_id"], group, row["media_id"], "complete", self._now()),
                )
                self.connection.execute(
                    "INSERT OR IGNORE INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"],
                        row["family_id"],
                        group,
                        row["media_type"],
                        row["asset_type"],
                        row["media_id"],
                        row.get("title"),
                        row.get("original_title"),
                        row.get("release_date"),
                        row["language"],
                        image,
                        row["tmdb_path"],
                        row.get("width") or 0,
                        row.get("height") or 0,
                        row.get("match_phash_distance"),
                        row.get("split"),
                        int(bool(row.get("all_languages_mined"))),
                        row.get("collected_at_utc") or self._now(),
                    ),
                )
        marker.write_text(self._now())

    def _write_state(self, group, page, achieved, target, exhausted, reason):
        with self.connection:
            self.connection.execute(
                """INSERT INTO mining_state VALUES(?,?,?,?,?,?,?,?,?)
              ON CONFLICT(group_name,source_language,target_language) DO UPDATE SET next_page=excluded.next_page,
              achieved_families=excluded.achieved_families,target_families=excluded.target_families,
              exhausted=excluded.exhausted,exhaustion_reason=excluded.exhaustion_reason,
              updated_at_utc=excluded.updated_at_utc""",
                (
                    group,
                    self.config.source_language,
                    self.config.target_language,
                    page,
                    achieved,
                    target,
                    int(exhausted),
                    reason,
                    self._now(),
                ),
            )

    def _api_get(self, path, params=None):
        params, headers = dict(params or {}), {"Accept": "application/json"}
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        else:
            raise RuntimeError("TMDB credentials are required for API requests")
        for attempt in range(8):
            wait = 1 / self.config.requests_per_second - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            response = self.session.get(self.API_URL + path, params=params, headers=headers, timeout=30)
            self._last_request = time.monotonic()
            if response.status_code == 429:
                time.sleep(max(float(response.headers.get("Retry-After", 1)), 2**attempt))
                continue
            if response.status_code >= 500:
                time.sleep(min(2**attempt, 30))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"TMDB request failed: {path}")

    def _download(self, path, size):
        response = self.session.get(f"{self.IMAGE_URL}/{size}{path}", timeout=60)
        if response.status_code == 404 and size != "original":
            response = self.session.get(f"{self.IMAGE_URL}/original{path}", timeout=60)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _notify(self, message):
        if not self.telegram_bot_token or self.telegram_chat_id in (None, ""):
            return False
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage",
                json={"chat_id": self.telegram_chat_id, "text": message[:4096]},
                timeout=30,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as error:
            warnings.warn(f"Telegram notification failed: {error}", RuntimeWarning, stacklevel=2)
            return False

    def _hf_record(self, row):
        row["group"] = row.pop("group_name")
        image_path = self.root / row.pop("image_path")
        if not image_path.is_file():
            raise FileNotFoundError(f"asset image is missing: {image_path}")
        row["image"] = {
            "bytes": image_path.read_bytes(),
            "path": image_path.name,
        }
        row["all_languages_mined"] = bool(row["all_languages_mined"])
        return row

    def _read_sql(self, sql, parameters=()):
        rows = self.connection.execute(sql, parameters).fetchall()
        return pl.DataFrame([dict(x) for x in rows]) if rows else pl.DataFrame()

    def _asset_count(self):
        return self.connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    def _family_count_all(self):
        return self.connection.execute("SELECT COUNT(*) FROM families WHERE status='complete'").fetchone()[0]

    def _completed_family_count(self, group):
        return self.connection.execute(
            "SELECT COUNT(*) FROM families WHERE group_name=? AND status='complete'", (group,)
        ).fetchone()[0]

    def _has_complete_family(self, group, media_id):
        return (
            self.connection.execute(
                "SELECT 1 FROM families WHERE group_name=? AND media_id=? AND status='complete' LIMIT 1",
                (group, media_id),
            ).fetchone()
            is not None
        )

    def _image_path(self, asset_id):
        return self.images_dir / asset_id[:2] / f"{asset_id}.jpg"

    def _cleanup_staging(self):
        for child in self.staging_dir.iterdir():
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)

    @staticmethod
    def _asset_id(group, media_id, path):
        return hashlib.sha256(f"tmdb-asset-v1\x1f{group}\x1f{media_id}\x1f{path}".encode()).hexdigest()[:24]

    @staticmethod
    def _family_id(group, media_id):
        return hashlib.sha256(f"tmdb-family-v2\x1f{group}\x1f{media_id}".encode()).hexdigest()[:24]

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _resolve_groups(cls, groups):
        if groups == "all":
            return tuple(cls.GROUPS)
        result = (groups,) if isinstance(groups, str) else tuple(groups)
        unknown = set(result) - set(cls.GROUPS)
        if unknown:
            raise ValueError(f"Unknown groups: {sorted(unknown)}")
        return tuple(dict.fromkeys(result))


TMDBPairMiner = TMDBAssetMiner
