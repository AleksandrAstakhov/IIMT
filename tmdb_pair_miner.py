"""Compatibility import for the asset-centric TMDB miner.

Deprecated module name: new code should import from ``tmdb_asset_miner``.
"""

from tmdb_asset_miner import MinerConfig, TMDBAssetMiner

# Preserve older notebooks without retaining the obsolete A/B storage schema.
TMDBPairMiner = TMDBAssetMiner

__all__ = ["MinerConfig", "TMDBAssetMiner", "TMDBPairMiner"]
