"""Backward-compatible entry point for the structured miner CLI."""

from iimt.miner.cli import as_list, load_env, main

__all__ = ["as_list", "load_env", "main"]


if __name__ == "__main__":
    main()
