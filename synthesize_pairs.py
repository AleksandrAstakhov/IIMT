"""Backward-compatible entry point for the synthetic dataset pipeline."""

from iimt.synthetic import pipeline as _pipeline
from iimt.synthetic.cli import main
from iimt.synthetic.pipeline import (
    Curve,
    prepare_massive,
    push_to_hub,
    synthesize,
    validate_dataset,
)

# Private aliases preserve compatibility with the former single-file module.
_assign_split = _pipeline._assign_split
_resolve_hf_repo = _pipeline._resolve_hf_repo
_resolve_hf_token = _pipeline._resolve_hf_token
_shared_masks = _pipeline._shared_masks
_write_pairs = _pipeline._write_pairs

__all__ = [
    "Curve",
    "prepare_massive",
    "push_to_hub",
    "synthesize",
    "validate_dataset",
]


if __name__ == "__main__":
    main()
