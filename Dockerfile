FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY iimt ./iimt
COPY tmdb_asset_miner.py run_miner.py synthesize_pairs.py ./

RUN pip install .

ENTRYPOINT ["tmdb-iimt-mine"]
