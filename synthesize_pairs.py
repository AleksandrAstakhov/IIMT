"""Render multilingual text variants on exactly the same curved baseline."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class Curve:
    """Cubic Bezier curve in image coordinates."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    p2: tuple[float, float]
    p3: tuple[float, float]

    def point(self, t: float) -> tuple[float, float]:
        u = 1.0 - t
        points = (self.p0, self.p1, self.p2, self.p3)
        weights = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
        return tuple(sum(w * p[axis] for w, p in zip(weights, points, strict=True)) for axis in (0, 1))

    def tangent(self, t: float) -> tuple[float, float]:
        u = 1.0 - t
        dx = 3 * u * u * (self.p1[0] - self.p0[0]) + 6 * u * t * (self.p2[0] - self.p1[0]) + 3 * t * t * (self.p3[0] - self.p2[0])
        dy = 3 * u * u * (self.p1[1] - self.p0[1]) + 6 * u * t * (self.p2[1] - self.p1[1]) + 3 * t * t * (self.p3[1] - self.p2[1])
        return dx, dy


def _seed(*parts: object) -> int:
    value = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _fonts(configured: list[str]) -> list[Path]:
    candidates = [Path(p).expanduser() for p in configured]
    if not candidates:
        roots = [Path("/usr/share/fonts"), Path("/System/Library/Fonts"), Path.home() / "Library/Fonts"]
        for root in roots:
            if root.exists():
                candidates.extend(root.rglob("*.ttf"))
                candidates.extend(root.rglob("*.otf"))
    valid = sorted({p.resolve() for p in candidates if p.is_file()})
    if not valid:
        raise RuntimeError("No TTF/OTF font found; set generation.font_paths=[/path/font.ttf]")
    return valid


def _supports(font_path: Path, text: str) -> bool:
    font = ImageFont.truetype(str(font_path), 32)
    missing = bytes(font.getmask("\ufffd"))
    return all(ch.isspace() or bytes(font.getmask(ch)) != missing for ch in text)


def _pick_font(paths: list[Path], text: str, start: int) -> Path:
    for offset in range(len(paths)):
        candidate = paths[(start + offset) % len(paths)]
        if _supports(candidate, text):
            return candidate
    raise RuntimeError(f"No configured font supports all characters in {text!r}")


def _requires_complex_shaping(text: str) -> bool:
    """Scripts that cannot be rendered correctly one Unicode code point at a time."""
    shaping_ranges = (
        (0x0590, 0x08FF),  # Hebrew, Arabic and related RTL scripts
        (0x0900, 0x0E7F),  # Indic and Thai scripts
        (0x0F00, 0x109F),  # Tibetan and Myanmar
        (0x1780, 0x17FF),  # Khmer
    )
    return any(
        unicodedata.bidirectional(character) in {"R", "AL", "AN"}
        or any(start <= ord(character) <= end for start, end in shaping_ranges)
        for character in text
    )


def _curve(rng: random.Random, width: int, height: int, margin: int, amp_min: int, amp_max: int) -> Curve:
    base = rng.uniform(height * 0.38, height * 0.68)
    amplitude = rng.choice((-1, 1)) * rng.uniform(amp_min, amp_max)
    return Curve(
        (margin, base),
        (width * 0.33, base - amplitude),
        (width * 0.67, base - amplitude),
        (width - margin, base),
    )


def _background(path: Path | None, width: int, height: int, rng: random.Random) -> Image.Image:
    if path:
        image = Image.open(path).convert("RGB")
        scale = max(width / image.width, height / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))
    c0 = tuple(rng.randint(20, 170) for _ in range(3))
    c1 = tuple(rng.randint(80, 235) for _ in range(3))
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        a = y / max(1, height - 1)
        color = tuple(round((1 - a) * x + a * z) for x, z in zip(c0, c1, strict=True))
        for x in range(width):
            pixels[x, y] = color
    return image


def _arc_table(curve: Curve, samples: int = 1000) -> tuple[list[float], list[float]]:
    ts = [i / samples for i in range(samples + 1)]
    lengths = [0.0]
    previous = curve.point(0.0)
    for t in ts[1:]:
        current = curve.point(t)
        lengths.append(lengths[-1] + math.dist(previous, current))
        previous = current
    return ts, lengths


def _t_at_length(ts: list[float], lengths: list[float], target: float) -> float:
    lo, hi = 0, len(lengths) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if lengths[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    i = max(1, lo)
    span = max(1e-8, lengths[i] - lengths[i - 1])
    a = (target - lengths[i - 1]) / span
    return ts[i - 1] + a * (ts[i] - ts[i - 1])


def _fit_font(path: Path, text: str, available: float, minimum: int, maximum: int) -> ImageFont.FreeTypeFont:
    for size in range(maximum, minimum - 1, -1):
        font = ImageFont.truetype(str(path), size)
        if sum(font.getlength(ch) for ch in text) <= available:
            return font
    return ImageFont.truetype(str(path), minimum)


def render_curved_text(
    background: Image.Image,
    text: str,
    curve: Curve,
    font_path: Path,
    min_font_size: int,
    max_font_size: int,
    stroke_width: int,
    shadow_offset: int,
) -> tuple[Image.Image, Image.Image, int]:
    """Place glyph centres by arc length and rotate each glyph to the shared tangent."""
    ts, lengths = _arc_table(curve)
    font = _fit_font(font_path, text, lengths[-1] * 0.92, min_font_size, max_font_size)
    advances = [font.getlength(ch) for ch in text]
    total = sum(advances)
    cursor = max(0.0, (lengths[-1] - total) / 2)
    text_layer = Image.new("RGBA", background.size)
    mask = Image.new("L", background.size)

    for ch, advance in zip(text, advances, strict=True):
        centre = cursor + advance / 2
        cursor += advance
        if ch.isspace():
            continue
        t = _t_at_length(ts, lengths, centre)
        x, y = curve.point(t)
        dx, dy = curve.tangent(t)
        angle = math.degrees(math.atan2(dy, dx))
        box = font.getbbox(ch, stroke_width=stroke_width)
        gw, gh = box[2] - box[0] + 12, box[3] - box[1] + 12

        glyph = Image.new("RGBA", (gw, gh))
        gd = ImageDraw.Draw(glyph)
        origin = (6 - box[0], 6 - box[1])
        gd.text((origin[0] + shadow_offset, origin[1] + shadow_offset), ch, font=font, fill=(0, 0, 0, 130), stroke_width=stroke_width)
        gd.text(origin, ch, font=font, fill=(250, 250, 245, 255), stroke_width=stroke_width, stroke_fill=(20, 20, 20, 255))
        rotated = glyph.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        pos = (round(x - rotated.width / 2), round(y - rotated.height / 2))
        text_layer.alpha_composite(rotated, pos)

        glyph_mask = Image.new("L", (gw, gh))
        ImageDraw.Draw(glyph_mask).text(origin, ch, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255)
        rotated_mask = glyph_mask.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        existing = mask.crop((pos[0], pos[1], pos[0] + rotated.width, pos[1] + rotated.height))
        mask.paste(ImageChops.lighter(existing, rotated_mask), pos)

    composed = Image.alpha_composite(background.convert("RGBA"), text_layer).convert("RGB")
    return composed, mask.filter(ImageFilter.MaxFilter(3)), font.size


def _shared_masks(curve: Curve, size: tuple[int, int], region_width: int) -> tuple[Image.Image, Image.Image]:
    """Create language-independent layout supervision from the structural curve."""
    points = [tuple(round(v) for v in curve.point(i / 512)) for i in range(513)]
    region = Image.new("L", size)
    baseline = Image.new("L", size)
    ImageDraw.Draw(region).line(points, fill=255, width=region_width, joint="curve")
    ImageDraw.Draw(baseline).line(points, fill=255, width=3, joint="curve")
    return region, baseline


def _assign_split(family_id: str, config: DictConfig) -> str:
    train = float(config.split.train)
    validation = float(config.split.validation)
    test = float(config.split.test)
    if not math.isclose(train + validation + test, 1.0, abs_tol=1e-6):
        raise ValueError("split.train + split.validation + split.test must equal 1")
    value = _seed(config.split.seed, family_id) / 2**64
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def _write_pairs(stream: Any, assets: list[dict[str, Any]], config: DictConfig) -> int:
    by_language = {row["language"]: row for row in assets}
    mode = str(config.pairing.mode)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    if mode == "all":
        pairs = list(itertools.permutations(assets, 2))
        if not bool(config.pairing.bidirectional):
            pairs = list(itertools.combinations(assets, 2))
    elif mode == "pivot":
        pivot = next((by_language[x] for x in config.pairing.pivot_languages if x in by_language), None)
        if pivot is None:
            return 0
        pairs = [(pivot, target) for target in assets if target["id"] != pivot["id"]]
        if bool(config.pairing.bidirectional):
            pairs += [(target, pivot) for target in assets if target["id"] != pivot["id"]]
    else:
        raise ValueError("pairing.mode must be 'pivot' or 'all'")

    for source, target in pairs:
        pair_id = hashlib.sha256(f"{source['id']}\0{target['id']}".encode()).hexdigest()[:24]
        stream.write(json.dumps({
            "id": pair_id,
            "family_id": source["family_id"],
            "split": source["split"],
            "source_id": source["id"],
            "target_id": target["id"],
            "source_language": source["language"],
            "target_language": target["language"],
            "source_text": source["text"],
            "target_text": target["text"],
        }, ensure_ascii=False) + "\n")
    return len(pairs)


def synthesize(config: DictConfig) -> Path:
    input_path = Path(config.input.translations)
    output_dir = Path(config.output.dir)
    directories = {
        "image": output_dir / "images",
        "glyph_mask": output_dir / "glyph_masks",
        "region_mask": output_dir / "region_masks",
        "baseline_mask": output_dir / "baseline_masks",
        "background": output_dir / "backgrounds",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    backgrounds_dir = Path(config.input.backgrounds_dir)
    backgrounds = sorted(p for p in backgrounds_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if backgrounds_dir.exists() else []
    font_paths = _fonts(list(config.generation.font_paths))
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    pairs_temporary = pairs_manifest.with_suffix(pairs_manifest.suffix + ".tmp")
    counts = Counter()

    with (
        input_path.open("r", encoding="utf-8") as records,
        temporary.open("w", encoding="utf-8") as asset_stream,
        pairs_temporary.open("w", encoding="utf-8") as pair_stream,
    ):
        for line in records:
            if not line.strip():
                continue
            record = json.loads(line)
            base_family = str(record["family_id"])
            translations = {}
            for language, text_value in record["translations"].items():
                text = str(text_value)
                if _requires_complex_shaping(text):
                    policy = str(config.generation.unsupported_script_policy)
                    if policy == "error":
                        raise ValueError(
                            f"{base_family}/{language} requires HarfBuzz/RTL shaping; "
                            "use a shaped renderer or unsupported_script_policy=skip"
                        )
                    if policy != "skip":
                        raise ValueError("unsupported_script_policy must be 'skip' or 'error'")
                    counts["skipped_complex_script_assets"] += int(config.generation.variants_per_family)
                    continue
                translations[language] = text
            if len(translations) < 2:
                counts["skipped_families"] += 1
                continue
            split = str(record.get("split") or _assign_split(base_family, config))
            if split == "dev":
                split = "validation"

            for variant in range(int(config.generation.variants_per_family)):
                family_id = f"{base_family}-v{variant:03d}"
                rng = random.Random(_seed(config.generation.seed, family_id))
                source_background = backgrounds[_seed(family_id, "background") % len(backgrounds)] if backgrounds else None
                background = _background(source_background, config.generation.width, config.generation.height, rng)
                curve = _curve(rng, config.generation.width, config.generation.height, config.generation.margin, config.generation.curve_amplitude_min, config.generation.curve_amplitude_max)
                region, baseline = _shared_masks(curve, background.size, int(config.generation.region_width))
                shared_id = hashlib.sha256(family_id.encode()).hexdigest()[:24]
                background_path = directories["background"] / f"{shared_id}.jpg"
                region_path = directories["region_mask"] / f"{shared_id}.png"
                baseline_path = directories["baseline_mask"] / f"{shared_id}.png"
                background.save(background_path, quality=95, subsampling=0)
                region.save(region_path)
                baseline.save(baseline_path)
                font_start = _seed(family_id, "font") % len(font_paths)
                family_assets: list[dict[str, Any]] = []

                for language, text_value in sorted(translations.items()):
                    text = str(text_value)
                    font_path = _pick_font(font_paths, text, font_start)
                    image, glyph_mask, font_size = render_curved_text(background, text, curve, font_path, config.generation.min_font_size, config.generation.max_font_size, config.generation.stroke_width, config.generation.shadow_offset)
                    asset_id = hashlib.sha256(f"{family_id}\0{language}".encode()).hexdigest()[:24]
                    image_path = directories["image"] / f"{asset_id}.jpg"
                    glyph_path = directories["glyph_mask"] / f"{asset_id}.png"
                    image.save(image_path, quality=95, subsampling=0)
                    glyph_mask.save(glyph_path)
                    row = {
                        "id": asset_id,
                        "base_family_id": base_family,
                        "family_id": family_id,
                        "variant": variant,
                        "split": split,
                        "language": language,
                        "text": text,
                        "image": image_path.relative_to(output_dir).as_posix(),
                        "glyph_mask": glyph_path.relative_to(output_dir).as_posix(),
                        "region_mask": region_path.relative_to(output_dir).as_posix(),
                        "baseline_mask": baseline_path.relative_to(output_dir).as_posix(),
                        "background": background_path.relative_to(output_dir).as_posix(),
                        "font": str(font_path),
                        "font_size": font_size,
                        "curve": asdict(curve),
                        "geometry_seed": _seed(config.generation.seed, family_id) & ((1 << 63) - 1),
                        "width": background.width,
                        "height": background.height,
                    }
                    asset_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    family_assets.append(row)
                    counts["assets"] += 1
                    counts[f"assets/{split}"] += 1
                counts["pairs"] += _write_pairs(pair_stream, family_assets, config)
                counts["families"] += 1

    temporary.replace(manifest)
    pairs_temporary.replace(pairs_manifest)
    print(json.dumps({**counts, "manifest": str(manifest), "pairs_manifest": str(pairs_manifest)}, ensure_ascii=False, indent=2))
    return manifest


def validate_dataset(config: DictConfig) -> dict[str, int]:
    """Fail fast on leakage, broken files, dimensions, or invalid supervision masks."""
    output_dir = Path(config.output.dir)
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    assets: dict[str, dict[str, Any]] = {}
    base_splits: dict[str, str] = {}
    family_languages: defaultdict[str, set[str]] = defaultdict(set)
    family_curves: dict[str, str] = {}
    counts = Counter()

    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["id"] in assets:
                raise ValueError(f"duplicate asset id at line {line_number}: {row['id']}")
            previous_split = base_splits.setdefault(row["base_family_id"], row["split"])
            if previous_split != row["split"]:
                raise ValueError(f"split leakage for {row['base_family_id']}")
            curve_json = json.dumps(row["curve"], sort_keys=True)
            if family_curves.setdefault(row["family_id"], curve_json) != curve_json:
                raise ValueError(f"inconsistent curve in {row['family_id']}")
            family_languages[row["family_id"]].add(row["language"])

            images = {name: Image.open(output_dir / row[name]) for name in (
                "image", "glyph_mask", "region_mask", "baseline_mask", "background"
            )}
            expected = (row["width"], row["height"])
            if any(image.size != expected for image in images.values()):
                raise ValueError(f"dimension mismatch for {row['id']}")
            outside = ImageChops.subtract(images["glyph_mask"].convert("L"), images["region_mask"].convert("L"))
            if outside.getbbox() is not None:
                raise ValueError(f"glyph mask escapes region mask for {row['id']}")
            if images["glyph_mask"].getbbox() is None or images["region_mask"].getbbox() is None:
                raise ValueError(f"empty mask for {row['id']}")
            assets[row["id"]] = row
            counts["assets"] += 1

    if any(len(languages) < 2 for languages in family_languages.values()):
        raise ValueError("every family must contain at least two languages")

    pair_ids: set[str] = set()
    with pairs_manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            pair = json.loads(line)
            if pair["id"] in pair_ids:
                raise ValueError(f"duplicate pair id at line {line_number}: {pair['id']}")
            source, target = assets.get(pair["source_id"]), assets.get(pair["target_id"])
            if source is None or target is None:
                raise ValueError(f"dangling pair at line {line_number}")
            if source["family_id"] != target["family_id"] or source["split"] != target["split"]:
                raise ValueError(f"cross-family or cross-split pair at line {line_number}")
            pair_ids.add(pair["id"])
            counts["pairs"] += 1

    counts["families"] = len(family_languages)
    counts["base_families"] = len(base_splits)
    print(json.dumps({"validation": "ok", **counts}, indent=2))
    return dict(counts)


def _asset_generator(manifest: str, output_dir: str, split: str):
    root = Path(output_dir).resolve()
    image_fields = {"image", "glyph_mask", "region_mask", "baseline_mask", "background"}
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != split:
                continue
            for field in image_fields:
                row[field] = str(root / row[field])
            row["curve"] = json.dumps(row["curve"], sort_keys=True)
            yield row


def _pair_generator(manifest: str, split: str):
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == split:
                yield row


def _available_splits(manifest: Path) -> list[str]:
    found: set[str] = set()
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                found.add(json.loads(line)["split"])
    return [split for split in ("train", "validation", "test") if split in found]


def push_to_hub(config: DictConfig) -> None:
    """Publish relational assets/pairs configs without duplicating image bytes in pair rows."""
    from datasets import Dataset, DatasetDict, Features, Value
    from datasets import Image as HFImage
    from huggingface_hub import HfApi

    repo_id = config.hf.repo
    token = config.hf.token or os.environ.get("HF_TOKEN")
    if not repo_id:
        raise ValueError("Set hf.repo=owner/dataset or HF_DATASET_REPO")
    if not token:
        raise ValueError("Set HF_TOKEN in the environment")
    output_dir = Path(config.output.dir).resolve()
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    image = HFImage()
    asset_features = Features({
        "id": Value("string"), "base_family_id": Value("string"), "family_id": Value("string"),
        "variant": Value("int32"), "split": Value("string"), "language": Value("string"),
        "text": Value("string"), "image": image, "glyph_mask": image, "region_mask": image,
        "baseline_mask": image, "background": image, "font": Value("string"),
        "font_size": Value("int32"), "curve": Value("string"), "geometry_seed": Value("int64"),
        "width": Value("int32"), "height": Value("int32"),
    })
    pair_features = Features({name: Value("string") for name in (
        "id", "family_id", "split", "source_id", "target_id", "source_language",
        "target_language", "source_text", "target_text"
    )})
    asset_splits = _available_splits(manifest)
    pair_splits = _available_splits(pairs_manifest)
    assets = DatasetDict({split: Dataset.from_generator(
        _asset_generator,
        gen_kwargs={"manifest": str(manifest), "output_dir": str(output_dir), "split": split},
        features=asset_features,
    ) for split in asset_splits})
    pairs = DatasetDict({split: Dataset.from_generator(
        _pair_generator,
        gen_kwargs={"manifest": str(pairs_manifest), "split": split},
        features=pair_features,
    ) for split in pair_splits})
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=bool(config.hf.private), exist_ok=True)
    assets.push_to_hub(repo_id, config_name="assets", token=token, private=bool(config.hf.private), max_shard_size=str(config.hf.max_shard_size))
    pairs.push_to_hub(repo_id, config_name="pairs", token=token, private=bool(config.hf.private), max_shard_size=str(config.hf.max_shard_size))
    print(f"Published https://huggingface.co/datasets/{repo_id} (configs: assets, pairs)")


@hydra.main(version_base=None, config_path="pkg://tmdb_iimt_miner.conf", config_name="synthesize")
def main(config: DictConfig) -> None:
    actions = {
        "synthesize": lambda: synthesize(config),
        "validate": lambda: validate_dataset(config),
        "push": lambda: push_to_hub(config),
    }
    for step in config.pipeline.steps:
        if step not in actions:
            raise ValueError(f"unknown pipeline step: {step}")
        actions[step]()


if __name__ == "__main__":
    main()
