# TMDB IIMT Miner

Возобновляемый сборщик мультиязычных постеров для экспериментов по in-image
machine translation. Майнер ищет визуально совпадающие локализации в TMDB,
группирует их по `family_id`, хранит состояние в SQLite и публикует датасет
в Hugging Face без постоянной локальной копии Parquet.

## Основные свойства

- одна строка соответствует одному уникальному изображению;
- все языковые версии объединяются через `family_id`;
- `PRIMARY KEY` и `UNIQUE` в SQLite предотвращают дубли;
- после остановки повторный запуск продолжает сохранённый датасет;
- изображения распределяются по поддиректориям по префиксу ID;
- train/validation/test делятся по `family_id`, без утечки между сплитами;
- Hugging Face Parquet-шарды создаются по одному и удаляются после загрузки;
- загрузка на HF возобновляется по таблице `exports`;
- Telegram сообщает о старте, прогрессе, ошибке и завершении.

## Структура репозитория

```text
.
├── tmdb_asset_miner.py       # майнер, SQLite и HF-export
├── tmdb_pair_miner.py        # совместимость со старым импортом
├── run_miner.py              # Hydra entrypoint: mine → split → push
├── tmdb_iimt_miner/
│   └── conf/miner.yaml       # основной Hydra-конфиг
├── pyproject.toml            # пакет и зависимости для uv/pip
├── .env.example              # шаблон конфигурации
├── Dockerfile
└── tests/
    ├── test_cli.py
    └── test_storage.py
```

Зависимости описаны в `pyproject.toml`; `uv.lock` фиксирует их версии для uv.
Отдельный `requirements.txt` не требуется: pip также читает `pyproject.toml`.

## Установка через uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
cp .env.example .env
```

Или через стандартный pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

## Конфигурация

Заполните `.env`:

```dotenv
TMDB_API_KEY=your_tmdb_api_key

HF_TOKEN=hf_token_with_write_access
HF_DATASET_REPO=username/tmdb-iimt

TELEGRAM_BOT_TOKEN=telegram_bot_token
TELEGRAM_CHAT_ID=123456789

DATASET_DIR=tmdb_dataset
TARGET_FAMILIES=1000
REQUESTS_PER_SECOND=8
TELEGRAM_PROGRESS_EVERY=100
```

Вместо `TMDB_API_KEY` можно указать `TMDB_BEARER_TOKEN`. Telegram-поля
необязательны. Токены не передавайте в аргументах командной строки и не
добавляйте `.env` в Git.

## Запуск

Полный pipeline:

```bash
uv run tmdb-iimt-mine
```

Эквивалент без установки CLI:

```bash
uv run python run_miner.py
```

Только сбор данных:

```bash
uv run tmdb-iimt-mine \
  'pipeline.steps=[mine]' \
  miner.target_families=10000
```

Только разбиение и публикация:

```bash
uv run tmdb-iimt-mine 'pipeline.steps=[split,push]'
```

Публичный HF-репозиторий:

```bash
uv run tmdb-iimt-mine hf.private=false
```

По умолчанию майнятся только релевантные для основного IIMT-эксперимента
группы:

```text
movie.poster
tv.poster
```

## Локальное хранилище

```text
tmdb_dataset/
├── catalog.sqlite
├── catalog.sqlite-wal
├── catalog.sqlite-shm
├── images/
│   ├── 00/
│   ├── 01/
│   └── ...
├── staging/
└── export/
```

`catalog.sqlite` является источником истины. Постоянный локальный Parquet не
создаётся. Если в каталоге найден старый `assets.parquet`, он однократно
импортируется в SQLite и сохраняется как резервная копия.

## HF-схема

Каждая строка содержит:

```text
id, family_id, group, media_type, asset_type, media_id,
title, original_title, release_date, language, image,
tmdb_path, width, height, match_phash_distance, split,
all_languages_mined, collected_at_utc
```

`image` публикуется как `datasets.Image`. Направленные пары создаются
self-join по `family_id` и выбранным языкам.

## Тесты

```bash
uv run pytest
uv run ruff check .
```

Тесты не обращаются к TMDB, Telegram или Hugging Face.

## Docker

```bash
docker build -t tmdb-iimt-miner .
docker run --rm \
  --env-file .env \
  -v "$PWD/tmdb_dataset:/app/tmdb_dataset" \
  tmdb-iimt-miner
```

Каталог с SQLite и изображениями обязательно должен находиться на постоянном
диске или volume.

## Synthetic IIMT dataset

Независимый Hydra-pipeline создаёт мультиязычные варианты на общей Bezier-кривой,
проверяет датасет и при необходимости публикует его в Hugging Face.

```bash
uv sync --extra dev
uv run iimt-synthesize
```

Собственные мультиязычные группы передаются в JSONL:

```json
{"family_id":"example-1","translations":{"en":"Summer sale","ru":"Летняя распродажа"}}
```

```bash
uv run iimt-synthesize \
  input.translations=/data/translations.jsonl \
  input.backgrounds_dir=/data/backgrounds \
  output.dir=synthetic_dataset \
  generation.variants_per_family=3
```

Выход содержит:

```text
synthetic_dataset/
├── assets.jsonl
├── pairs.jsonl
├── images/
├── backgrounds/
├── glyph_masks/
├── region_masks/
└── baseline_masks/
```

Split назначается по исходному `base_family_id`, поэтому языки и варианты одной
фразы не пересекаются между train, validation и test. `validate` проверяет ID,
файлы, размеры, общую геометрию, маски, пары и split leakage.

Публикация создаёт две HF-конфигурации: `assets` с изображениями и supervision,
и `pairs` с лёгким индексом направленных пар:

```bash
export HF_TOKEN=hf_...
export HF_DATASET_REPO=username/iimt-synthetic

uv run iimt-synthesize \
  'pipeline.steps=[validate,push]' \
  output.dir=synthetic_dataset
```

Hydra-конфиг находится в `tmdb_iimt_miner/conf/synthesize.yaml`. Текущий renderer
поддерживает латиницу, кириллицу, греческий, CJK и корейский; сложные RTL/shaping
письменности по умолчанию пропускаются, чтобы не загрязнять обучающие данные.
