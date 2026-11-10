"""Hydra orchestration for the synthetic dataset pipeline."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .generation import synthesize
from .hub import push_to_hub
from .prepare import prepare_massive
from .validation import validate_dataset


@hydra.main(version_base=None, config_path="pkg://iimt.conf", config_name="synthesize")
def main(config: DictConfig) -> None:
    actions = {
        "prepare": lambda: prepare_massive(config),
        "synthesize": lambda: synthesize(config),
        "validate": lambda: validate_dataset(config),
        "push": lambda: push_to_hub(config),
    }
    for step in config.pipeline.steps:
        try:
            action = actions[str(step)]
        except KeyError as error:
            raise ValueError(f"unknown pipeline step: {step}") from error
        action()


if __name__ == "__main__":
    main()
