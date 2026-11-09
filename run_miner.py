#!/usr/bin/env python3
"""Hydra entry point for mine -> split -> Hugging Face push."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from tmdb_asset_miner import MinerConfig, TMDBAssetMiner


def load_env(path: Path) -> None:
    """Load a simple KEY=VALUE file without overriding process environment."""
    if not path.exists():
        return
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line {number} in {path}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"Invalid empty key on line {number} in {path}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def as_list(value: ListConfig | list | tuple) -> list[str]:
    return [str(item) for item in value]


@hydra.main(version_base=None, config_path="pkg://tmdb_iimt_miner.conf", config_name="miner")
def main(cfg: DictConfig) -> None:
    load_env(Path(to_absolute_path(str(cfg.env_file))))

    tmdb_api_key = os.getenv("TMDB_API_KEY")
    tmdb_bearer_token = os.getenv("TMDB_BEARER_TOKEN")
    if not tmdb_api_key and not tmdb_bearer_token:
        raise SystemExit("Set TMDB_API_KEY or TMDB_BEARER_TOKEN in .env")

    steps = as_list(cfg.pipeline.steps)
    groups = as_list(cfg.miner.groups)
    output_dir = Path(to_absolute_path(str(cfg.paths.output_dir)))
    hf_repo = str(cfg.hf.repo or os.getenv("HF_DATASET_REPO", "")).strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if "push" in steps and not hf_repo:
        raise SystemExit("Set HF_DATASET_REPO in .env or use hf.repo=username/dataset")
    if "push" in steps and not hf_token:
        raise SystemExit("Set HF_TOKEN in .env")

    miner_config = MinerConfig(
        output_dir=output_dir,
        groups=tuple(groups),
        source_language=str(cfg.miner.source_language),
        target_language=str(cfg.miner.target_language),
        max_candidates_per_language=int(cfg.miner.max_candidates_per_language),
        max_phash_distance=int(cfg.miner.max_phash_distance),
        requests_per_second=float(cfg.miner.requests_per_second),
        image_size=str(cfg.miner.image_size),
        candidate_image_size=str(cfg.miner.candidate_image_size),
        min_vote_count=int(cfg.miner.min_vote_count),
        max_discover_page=int(cfg.miner.max_discover_page),
        strict_target=bool(cfg.miner.strict_target),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        telegram_progress_every=int(cfg.telegram.progress_every),
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(f"Dataset directory: {output_dir}")
    if hf_repo:
        print(f"HF dataset: {hf_repo}")

    with TMDBAssetMiner(
        miner_config,
        api_key=tmdb_api_key,
        bearer_token=tmdb_bearer_token,
    ) as miner:
        try:
            if "mine" in steps:
                print("\n[mine]")
                print(miner.mine(target_families=int(cfg.miner.target_families)))

            if "split" in steps:
                print("\n[split]")
                print(miner.split(
                    train=float(cfg.split.train),
                    validation=float(cfg.split.validation),
                    test=float(cfg.split.test),
                    seed=int(cfg.split.seed),
                ))

            if "push" in steps:
                miner.notify(f"☁️ Hugging Face upload started\nDataset: {hf_repo}")
                print("\n[push]")
                miner.push(
                    repo_id=hf_repo,
                    token=hf_token,
                    private=bool(cfg.hf.private),
                    export_id=str(cfg.hf.export_id),
                    rows_per_shard=int(cfg.hf.rows_per_shard),
                )
                miner.notify(f"✅ Hugging Face upload completed\nDataset: {hf_repo}")
        except BaseException as error:
            miner.notify(
                "❌ Miner pipeline failed\n"
                f"Stages: {', '.join(steps)}\n"
                f"Error: {type(error).__name__}: {error}"
            )
            raise


if __name__ == "__main__":
    main()
