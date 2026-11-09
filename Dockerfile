FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tmdb_iimt_miner ./tmdb_iimt_miner
COPY tmdb_asset_miner.py tmdb_pair_miner.py run_miner.py synthesize_pairs.py ./

RUN pip install .

ENTRYPOINT ["tmdb-iimt-mine"]
