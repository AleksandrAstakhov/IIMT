"""Render multilingual text variants on exactly the same curved baseline."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import tarfile
import traceback
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
import requests
from fontTools.ttLib import TTFont
from omegaconf import DictConfig, OmegaConf
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
        x = sum(w * point[0] for w, point in zip(weights, points, strict=True))
        y = sum(w * point[1] for w, point in zip(weights, points, strict=True))
        return float(x), float(y)

    def tangent(self, t: float) -> tuple[float, float]:
        u = 1.0 - t
        dx = (
            3 * u * u * (self.p1[0] - self.p0[0])
            + 6 * u * t * (self.p2[0] - self.p1[0])
            + 3 * t * t * (self.p3[0] - self.p2[0])
        )
        dy = (
            3 * u * u * (self.p1[1] - self.p0[1])
            + 6 * u * t * (self.p2[1] - self.p1[1])
            + 3 * t * t * (self.p3[1] - self.p2[1])
        )
        return dx, dy


class TextDoesNotFitError(ValueError):
    """The rendered text would be clipped or cannot fit the structural curve."""


class FontNotFoundError(ValueError):
    """No configured font can render every character in a translation."""


@dataclass(frozen=True)
class FamilyRenderResult:
    """Outcome of rendering every supported language in one family."""

    family_id: str
    assets: list[dict[str, Any]]
    rejection: dict[str, Any] | None = None
    skipped_languages: tuple[str, ...] = ()


def _seed(*parts: object) -> int:
    value = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _fonts(configured: list[str], font_family: str) -> list[Path]:
    candidates = [Path(p).expanduser() for p in configured]
    if not candidates:
        roots = [Path("/usr/share/fonts"), Path("/System/Library/Fonts"), Path.home() / "Library/Fonts"]
        for root in roots:
            if root.exists():
                candidates.extend(root.rglob("*.ttf"))
                candidates.extend(root.rglob("*.otf"))
                candidates.extend(root.rglob("*.ttc"))
    valid = []
    for candidate in sorted({p.resolve() for p in candidates if p.is_file()}):
        try:
            ImageFont.truetype(str(candidate), 32)
        except OSError:
            continue
        valid.append(candidate)
    if not configured:
        family_key = font_family.lower().replace(" ", "").replace("-", "")
        valid = [
            path
            for path in valid
            if family_key in path.name.lower().replace(" ", "").replace("-", "")
            and "mono" not in path.name.lower()
        ]
    if not valid:
        raise RuntimeError(
            f"No TTF/OTF/TTC font found for family {font_family!r}; install it or set generation.font_paths"
        )
    return valid


def _style_fonts(paths: list[Path], style_level: int) -> list[Path]:
    """Select one weight from a shared font superfamily for a causal style value."""
    weight = "regular" if style_level % 2 == 0 else "bold"
    if weight == "regular":
        non_regular_tokens = ("bold", "black", "medium", "semibold", "italic")
        selected = [
            path
            for path in paths
            if "regular" in path.name.lower()
            or not any(token in path.name.lower() for token in non_regular_tokens)
        ]
    else:
        selected = [path for path in paths if "bold" in path.name.lower()]
    if not selected:
        raise FontNotFoundError(f"font superfamily has no {weight} face")
    return selected


def _text_palette(style_level: int) -> dict[str, tuple[int, int, int, int]]:
    palettes = (
        {
            "fill": (250, 250, 245, 255),
            "stroke": (20, 20, 20, 255),
            "shadow": (0, 0, 0, 130),
        },
        {
            "fill": (255, 213, 64, 255),
            "stroke": (42, 24, 8, 255),
            "shadow": (24, 8, 0, 150),
        },
    )
    return palettes[style_level % len(palettes)]


@lru_cache(maxsize=256)
def _font_codepoints(font_path: str) -> frozenset[int]:
    """Read the font cmap once per worker; this avoids unreliable tofu-glyph checks."""
    try:
        with TTFont(font_path, fontNumber=0, lazy=True) as font:
            return frozenset(font.getBestCmap() or {})
    except Exception:
        return frozenset()


def _supports(font_path: Path, text: str) -> bool:
    try:
        ImageFont.truetype(str(font_path), 32)
    except OSError:
        return False
    codepoints = _font_codepoints(str(font_path))
    return bool(codepoints) and all(ch.isspace() or ord(ch) in codepoints for ch in text)


def _pick_font(paths: list[Path], text: str, start: int) -> Path:
    for offset in range(len(paths)):
        candidate = paths[(start + offset) % len(paths)]
        if _supports(candidate, text):
            return candidate
    raise FontNotFoundError(f"No configured font supports all characters in {text!r}")


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


def _normalize_text(text: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def _compact_phrase(text: str, config: DictConfig) -> bool:
    if not text or len(text) > int(config.prepare.max_characters):
        return False
    lowered = text.lower()
    if any(token in lowered for token in ("http://", "https://", "www.")):
        return False
    words = text.split()
    if len(words) == 1:
        has_unspaced_script = any(
            0x3040 <= ord(character) <= 0x30FF  # Japanese Kana
            or 0x3400 <= ord(character) <= 0x9FFF  # CJK
            or 0xAC00 <= ord(character) <= 0xD7AF  # Hangul
            for character in text
        )
        return has_unspaced_script and 2 <= len(text) <= min(40, int(config.prepare.max_characters))
    return int(config.prepare.min_words) <= len(words) <= int(config.prepare.max_words)


def _download_file(url: str, destination: Path) -> None:
    """Download atomically so an interrupted request never looks complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if downloaded and response.status_code == 200:
            downloaded = 0
            temporary.unlink(missing_ok=True)
        response.raise_for_status()
        mode = "ab" if downloaded and response.status_code == 206 else "wb"
        expected = int(response.headers.get("content-length", 0)) + downloaded
        written = downloaded
        with temporary.open(mode) as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                written += len(chunk)
                if written // (100 * 1024 * 1024) != (written - len(chunk)) // (100 * 1024 * 1024):
                    print(f"downloaded {written / 1024**2:.0f} MiB")
    if expected and written != expected:
        raise OSError(f"incomplete download: expected {expected} bytes, got {written}")
    temporary.replace(destination)


def prepare_massive(config: DictConfig) -> Path:
    """Build compact multi-way translations directly from the official MASSIVE archive."""
    archive_path = Path(config.prepare.archive_path)
    output_path = Path(config.prepare.output_path)
    if output_path.exists() and not bool(config.prepare.force):
        print(f"prepare: reusing {output_path}; set prepare.force=true to rebuild")
        return output_path
    if not archive_path.exists():
        print(f"prepare: downloading MASSIVE 1.1 to {archive_path}")
        _download_file(str(config.prepare.url), archive_path)

    locale_value = config.prepare.locales
    selected_locales = (
        None if isinstance(locale_value, str) and locale_value.lower() == "all" else set(locale_value)
    )
    selected_partitions = set(config.prepare.partitions)
    accepted_by_locale: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_path = output_path.with_suffix(".flat.tmp.jsonl")

    with flat_path.open("w", encoding="utf-8") as flat_output:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.lower().endswith((".jsonl", ".json"))
            ]
            if not members:
                raise ValueError(f"no JSON/JSONL files found in {archive_path}")
            for member in members:
                source = archive.extractfile(member)
                if source is None:
                    continue
                for raw_line in source:
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line)
                    locale = str(row["locale"])
                    partition = str(row["partition"])
                    if selected_locales is not None and locale not in selected_locales:
                        continue
                    if partition not in selected_partitions:
                        continue
                    text = _normalize_text(row["utt"])
                    if not _compact_phrase(text, config):
                        continue
                    flat_output.write(
                        json.dumps(
                            {
                                "partition": partition,
                                "original_id": str(row["id"]),
                                "scenario": row.get("scenario"),
                                "intent": row.get("intent"),
                                "locale": locale,
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted_by_locale[locale] += 1

    if not accepted_by_locale:
        flat_path.unlink(missing_ok=True)
        raise ValueError("MASSIVE preprocessing produced no compact phrases")

    minimum = int(config.prepare.min_languages_per_group)
    frame = pl.read_ndjson(flat_path)
    grouped = (
        frame.sort(["partition", "original_id", "locale", "text"])
        .unique(subset=["partition", "original_id", "locale"], keep="first", maintain_order=True)
        .group_by(["partition", "original_id"], maintain_order=True)
        .agg(
            pl.first("scenario").alias("scenario"),
            pl.first("intent").alias("intent"),
            pl.struct(["locale", "text"]).alias("translations"),
            pl.len().alias("language_count"),
        )
        .filter(pl.col("language_count") >= minimum)
        .sort(["partition", "original_id"])
    )
    if config.prepare.max_groups is not None:
        grouped = grouped.head(int(config.prepare.max_groups))
    flat_path.unlink(missing_ok=True)
    if grouped.is_empty():
        raise ValueError("MASSIVE preprocessing produced no eligible multilingual groups")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    split_names = {"train": "train", "dev": "validation", "test": "test"}
    with temporary.open("w", encoding="utf-8") as output:
        for group in grouped.iter_rows(named=True):
            partition = group["partition"]
            original_id = group["original_id"]
            translations = {item["locale"]: item["text"] for item in group["translations"]}
            output.write(
                json.dumps(
                    {
                        "family_id": f"massive-{partition}-{original_id}",
                        "split": split_names.get(partition, partition),
                        "scenario": group["scenario"],
                        "intent": group["intent"],
                        "translations": dict(sorted(translations.items())),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "prepare": "ok",
                "groups": grouped.height,
                "languages": len(accepted_by_locale),
                "accepted_by_locale": dict(sorted(accepted_by_locale.items())),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_path


def _curve(rng: random.Random, width: int, height: int, margin: int, amp_min: int, amp_max: int) -> Curve:
    base = rng.uniform(height * 0.38, height * 0.68)
    amplitude = rng.choice((-1, 1)) * rng.uniform(amp_min, amp_max)
    return Curve(
        (margin, base),
        (width * 0.33, base - amplitude),
        (width * 0.67, base - amplitude),
        (width - margin, base),
    )


def _placement_candidate(generation: dict[str, int], family_id: str, attempt: int) -> tuple[Curve, int]:
    total = max(1, generation["placement_attempts"] - 1)
    progress = attempt / total
    rng = random.Random(_seed(generation["seed"], family_id, "placement", attempt))
    width, height = generation["width"], generation["height"]
    minimum_margin = generation["image_safe_margin"] + 12
    margin = round(generation["margin"] * (1.0 - progress) + minimum_margin * progress)
    vertical_span = 0.15 * (1.0 - progress)
    base = height * (0.5 + rng.uniform(-vertical_span, vertical_span))
    amplitude_scale = 1.0 - 0.85 * progress
    amp_min = generation["curve_amplitude_min"] * amplitude_scale
    amp_max = generation["curve_amplitude_max"] * amplitude_scale
    amplitude = rng.choice((-1, 1)) * rng.uniform(min(amp_min, amp_max), max(amp_min, amp_max))
    curve = Curve(
        (margin, base),
        (width * 0.33, base - amplitude),
        (width * 0.67, base - amplitude),
        (width - margin, base),
    )
    max_font_size = round(
        generation["max_font_size"] * (1.0 - progress) + generation["min_font_size"] * progress
    )
    return curve, max(generation["min_font_size"], max_font_size)


def _background(path: Path | None, width: int, height: int, rng: random.Random) -> Image.Image:
    if path:
        image = Image.open(path).convert("RGB")
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
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
    low, high = minimum, maximum
    best: ImageFont.FreeTypeFont | None = None
    while low <= high:
        size = (low + high) // 2
        font = ImageFont.truetype(str(path), size)
        if sum(font.getlength(ch) for ch in text) <= available:
            best = font
            low = size + 1
        else:
            high = size - 1
    if best is not None:
        return best
    raise TextDoesNotFitError(f"text is too long for the curve at minimum font size {minimum}: {text!r}")


def _preflight_curved_text(
    image_size: tuple[int, int],
    text: str,
    curve: Curve,
    font_path: Path,
    min_font_size: int,
    max_font_size: int,
    stroke_width: int,
    image_safe_margin: int,
) -> int:
    """Check exact glyph placement using metrics without allocating or rotating pixels."""
    ts, lengths = _arc_table(curve)
    font = _fit_font(font_path, text, lengths[-1] * 0.92, min_font_size, max_font_size)
    advances = [font.getlength(character) for character in text]
    cursor = max(0.0, (lengths[-1] - sum(advances)) / 2)
    image_width, image_height = image_size

    for character, advance in zip(text, advances, strict=True):
        centre = cursor + advance / 2
        cursor += advance
        if character.isspace():
            continue
        t = _t_at_length(ts, lengths, centre)
        x, y = curve.point(t)
        dx, dy = curve.tangent(t)
        angle = math.atan2(dy, dx)
        box = font.getbbox(character, stroke_width=stroke_width)
        glyph_width = box[2] - box[0] + 12
        glyph_height = box[3] - box[1] + 12
        rotated_width = (
            math.ceil(abs(glyph_width * math.cos(angle)) + abs(glyph_height * math.sin(angle))) + 4
        )
        rotated_height = (
            math.ceil(abs(glyph_width * math.sin(angle)) + abs(glyph_height * math.cos(angle))) + 4
        )
        if (
            x - rotated_width / 2 < image_safe_margin
            or y - rotated_height / 2 < image_safe_margin
            or x + rotated_width / 2 > image_width - image_safe_margin
            or y + rotated_height / 2 > image_height - image_safe_margin
        ):
            raise TextDoesNotFitError(f"glyph {character!r} would leave the image bounds in {text!r}")
    return font.size


def render_curved_text(
    background: Image.Image,
    text: str,
    curve: Curve,
    font_path: Path,
    min_font_size: int,
    max_font_size: int,
    stroke_width: int,
    shadow_offset: int,
    image_safe_margin: int,
    palette: dict[str, tuple[int, int, int, int]],
    fitted_font_size: int | None = None,
) -> tuple[Image.Image, Image.Image, int]:
    """Place glyph centres by arc length and rotate each glyph to the shared tangent."""
    ts, lengths = _arc_table(curve)
    font = (
        ImageFont.truetype(str(font_path), fitted_font_size)
        if fitted_font_size is not None
        else _fit_font(font_path, text, lengths[-1] * 0.92, min_font_size, max_font_size)
    )
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
        gd.text(
            (origin[0] + shadow_offset, origin[1] + shadow_offset),
            ch,
            font=font,
            fill=palette["shadow"],
            stroke_width=stroke_width,
        )
        gd.text(
            origin,
            ch,
            font=font,
            fill=palette["fill"],
            stroke_width=stroke_width,
            stroke_fill=palette["stroke"],
        )
        rotated = glyph.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
        pos = (round(x - rotated.width / 2), round(y - rotated.height / 2))
        if (
            pos[0] < image_safe_margin
            or pos[1] < image_safe_margin
            or pos[0] + rotated.width > background.width - image_safe_margin
            or pos[1] + rotated.height > background.height - image_safe_margin
        ):
            raise TextDoesNotFitError(f"glyph {ch!r} would leave the image bounds in {text!r}")
        text_layer.alpha_composite(rotated, pos)

        glyph_mask = Image.new("L", (gw, gh))
        ImageDraw.Draw(glyph_mask).text(
            origin, ch, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255
        )
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
        stream.write(
            json.dumps(
                {
                    "id": pair_id,
                    "pair_type": "translation",
                    "intervention_type": source["intervention_type"],
                    "changed_factor": "text",
                    "base_family_id": source["base_family_id"],
                    "family_id": source["family_id"],
                    "split": source["split"],
                    "source_id": source["id"],
                    "target_id": target["id"],
                    "source_language": source["language"],
                    "target_language": target["language"],
                    "source_text": source["text"],
                    "target_text": target["text"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return len(pairs)


def _write_causal_pairs(asset_manifest: Path, output_manifest: Path) -> int:
    """Pair reference and single-factor interventions for the same language."""

    def flush_family(stream: Any, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        reference = {row["language"]: row for row in rows if row["intervention_name"] == "reference"}
        written = 0
        for target in rows:
            if target["intervention_type"] != "intervention":
                continue
            source = reference.get(target["language"])
            if source is None:
                continue
            pair_id = hashlib.sha256(f"{source['id']}\0{target['id']}\0causal".encode()).hexdigest()[:24]
            stream.write(
                json.dumps(
                    {
                        "id": pair_id,
                        "pair_type": "causal_intervention",
                        "base_family_id": source["base_family_id"],
                        "split": source["split"],
                        "language": source["language"],
                        "text": source["text"],
                        "source_id": source["id"],
                        "target_id": target["id"],
                        "intervention_type": target["intervention_name"],
                        "changed_factor": target["changed_factor"],
                        "source_style_id": source["style_id"],
                        "target_style_id": target["style_id"],
                        "source_layout_id": source["layout_id"],
                        "target_layout_id": target["layout_id"],
                        "source_background_id": source["background_id"],
                        "target_background_id": target["background_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
        return written

    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    total = 0
    current_base: str | None = None
    family_rows: list[dict[str, Any]] = []
    with (
        asset_manifest.open("r", encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            row = json.loads(line)
            base_family = str(row["base_family_id"])
            if current_base is not None and base_family != current_base:
                total += flush_family(target, family_rows)
                family_rows = []
            current_base = base_family
            family_rows.append(row)
        total += flush_family(target, family_rows)
    temporary.replace(output_manifest)
    return total


_RENDER_CONTEXT: dict[str, Any] = {}


def _cgroup_cpu_count() -> int:
    """Return the effective container CPU quota instead of host os.cpu_count()."""
    candidates: list[int] = []
    host_count = os.cpu_count()
    if host_count:
        candidates.append(host_count)
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text(encoding="utf-8").strip().split()
        if quota != "max" and int(period) > 0:
            candidates.append(max(1, math.ceil(int(quota) / int(period))))
    else:
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.is_file() and period_path.is_file():
            quota_value = int(quota_path.read_text(encoding="utf-8").strip())
            period_value = int(period_path.read_text(encoding="utf-8").strip())
            if quota_value > 0 and period_value > 0:
                candidates.append(max(1, math.ceil(quota_value / period_value)))
    return min(candidates) if candidates else 1


def _init_render_worker(context: dict[str, Any]) -> None:
    global _RENDER_CONTEXT
    _RENDER_CONTEXT = context


def _render_family(
    task: tuple[str, dict[str, str], str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    base_family, translations, split, intervention = task
    context = _RENDER_CONTEXT
    generation = context["generation"]
    output_dir = Path(context["output_dir"])
    directories = {name: Path(path) for name, path in context["directories"].items()}
    backgrounds = [Path(path) for path in context["backgrounds"]]
    font_paths = [Path(path) for path in context["font_paths"]]
    intervention_name = str(intervention["name"])
    style_level = int(intervention["style_level"])
    layout_level = int(intervention["layout_level"])
    background_level = int(intervention["background_level"])
    family_id = f"{base_family}-{intervention_name}"
    style_id = hashlib.sha256(f"{base_family}\0style\0{style_level}".encode()).hexdigest()[:24]
    layout_id = hashlib.sha256(f"{base_family}\0layout\0{layout_level}".encode()).hexdigest()[:24]
    background_id = hashlib.sha256(f"{base_family}\0background\0{background_level}".encode()).hexdigest()[:24]
    background_rng = random.Random(_seed(generation["seed"], base_family, "background", background_level))
    source_background = (
        backgrounds[_seed(base_family, "background", background_level) % len(backgrounds)]
        if backgrounds
        else None
    )
    background = _background(
        source_background,
        generation["width"],
        generation["height"],
        background_rng,
    )
    shared_id = hashlib.sha256(family_id.encode()).hexdigest()[:24]
    background_path = directories["background"] / f"{shared_id}.jpg"
    region_path = directories["region_mask"] / f"{shared_id}.png"
    baseline_path = directories["baseline_mask"] / f"{shared_id}.png"
    style_font_paths = _style_fonts(font_paths, style_level)
    font_start = _seed(base_family, "font", style_level) % len(style_font_paths)
    palette = _text_palette(style_level)
    selected_fonts: dict[str, Path] = {}
    skipped_languages: list[str] = []
    for language, text in sorted(translations.items()):
        try:
            selected_fonts[language] = _pick_font(style_font_paths, text, font_start)
        except FontNotFoundError:
            skipped_languages.append(language)
    translations = {language: text for language, text in translations.items() if language in selected_fonts}
    if len(translations) < 2:
        raise FontNotFoundError(
            f"{family_id} has only {len(translations)} renderable language(s); "
            f"unsupported languages: {', '.join(skipped_languages)}"
        )
    rendered: list[tuple[str, str, Path, Image.Image, Image.Image, int]] | None = None
    curve: Curve | None = None
    placement_attempt = -1
    last_error: TextDoesNotFitError | None = None

    for attempt in range(generation["placement_attempts"]):
        candidate_curve, candidate_max_font = _placement_candidate(generation, layout_id, attempt)
        candidate_rendered: list[tuple[str, str, Path, Image.Image, Image.Image, int]] = []
        try:
            fitted_sizes = {
                language: _preflight_curved_text(
                    background.size,
                    text,
                    candidate_curve,
                    selected_fonts[language],
                    generation["min_font_size"],
                    candidate_max_font,
                    generation["stroke_width"],
                    generation["image_safe_margin"],
                )
                for language, text in sorted(translations.items())
            }
            for language, text in sorted(translations.items()):
                font_path = selected_fonts[language]
                image, glyph_mask, font_size = render_curved_text(
                    background,
                    text,
                    candidate_curve,
                    font_path,
                    generation["min_font_size"],
                    candidate_max_font,
                    generation["stroke_width"],
                    generation["shadow_offset"],
                    generation["image_safe_margin"],
                    palette,
                    fitted_sizes[language],
                )
                candidate_rendered.append((language, text, font_path, image, glyph_mask, font_size))
        except TextDoesNotFitError as error:
            last_error = error
            continue
        rendered = candidate_rendered
        curve = candidate_curve
        placement_attempt = attempt
        break

    if rendered is None or curve is None:
        raise TextDoesNotFitError(
            f"cannot place every translation of {family_id} after "
            f"{generation['placement_attempts']} attempts; last error: {last_error}"
        )

    region, baseline = _shared_masks(curve, background.size, generation["region_width"])
    background.save(background_path, quality=95, subsampling=0)
    baseline.save(baseline_path)
    family_assets: list[dict[str, Any]] = []
    required_region = Image.new("L", background.size)

    for language, text, font_path, image, glyph_mask, font_size in rendered:
        asset_id = hashlib.sha256(f"{family_id}\0{language}".encode()).hexdigest()[:24]
        image_path = directories["image"] / f"{asset_id}.jpg"
        glyph_path = directories["glyph_mask"] / f"{asset_id}.png"
        image.save(image_path, quality=95, subsampling=0)
        glyph_mask.save(glyph_path)
        padding = generation["region_padding"]
        padded_glyph = glyph_mask.filter(ImageFilter.MaxFilter(2 * padding + 1)) if padding else glyph_mask
        required_region = ImageChops.lighter(required_region, padded_glyph)
        family_assets.append(
            {
                "id": asset_id,
                "base_family_id": base_family,
                "family_id": family_id,
                "variant": int(intervention["index"]),
                "intervention_name": intervention_name,
                "intervention_type": str(intervention["intervention_type"]),
                "changed_factor": str(intervention["changed_factor"]),
                "style_id": style_id,
                "layout_id": layout_id,
                "background_id": background_id,
                "font_family": str(context["font_family"]),
                "style_level": style_level,
                "layout_level": layout_level,
                "background_level": background_level,
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
                "placement_attempt": placement_attempt,
                "curve": asdict(curve),
                "geometry_seed": _seed(generation["seed"], base_family, "layout", layout_level)
                & ((1 << 63) - 1),
                "width": background.width,
                "height": background.height,
            }
        )
    adaptive_width = generation["region_width"]
    width_step = max(2, 2 * generation["region_padding"])
    while ImageChops.subtract(required_region, region).getbbox() is not None:
        adaptive_width += width_step
        if adaptive_width > 2 * max(background.size):
            raise ValueError(f"cannot construct a region mask for {family_id}")
        region, _ = _shared_masks(curve, background.size, adaptive_width)
    region.save(region_path)
    return family_assets, tuple(skipped_languages)


def _render_family_task(
    task: tuple[str, dict[str, str], str, dict[str, Any]],
) -> FamilyRenderResult:
    """Render one complete family, converting sample-level failures to quarantine rows."""
    base_family, translations, split, intervention = task
    family_id = f"{base_family}-{intervention['name']}"
    try:
        assets, skipped_languages = _render_family(task)
        return FamilyRenderResult(
            family_id=family_id,
            assets=assets,
            skipped_languages=skipped_languages,
        )
    except (FontNotFoundError, TextDoesNotFitError) as error:
        reason = "missing_font" if isinstance(error, FontNotFoundError) else "placement_failed"
        return FamilyRenderResult(
            family_id=family_id,
            assets=[],
            rejection={
                "family_id": family_id,
                "base_family_id": base_family,
                "split": split,
                "languages": sorted(translations),
                "reason": reason,
                "error": str(error),
            },
        )
    except Exception as error:
        if not bool(_RENDER_CONTEXT["quality"]["quarantine_unexpected_errors"]):
            raise
        return FamilyRenderResult(
            family_id=family_id,
            assets=[],
            rejection={
                "family_id": family_id,
                "base_family_id": base_family,
                "split": split,
                "languages": sorted(translations),
                "reason": "unexpected_render_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            },
        )


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
    backgrounds = (
        sorted(
            p for p in backgrounds_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if backgrounds_dir.exists()
        else []
    )
    font_family = str(config.generation.font_family)
    font_paths = _fonts(list(config.generation.font_paths), font_family)
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    causal_pairs_manifest = output_dir / str(config.output.causal_pairs_manifest)
    rejected_manifest = output_dir / str(config.output.rejected_manifest)
    quality_report = output_dir / str(config.output.quality_report)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    pairs_temporary = pairs_manifest.with_suffix(pairs_manifest.suffix + ".tmp")
    rejected_temporary = rejected_manifest.with_suffix(rejected_manifest.suffix + ".tmp")
    counts: Counter[str] = Counter()
    raw_plan = OmegaConf.to_container(config.interventions.plan, resolve=True)
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError("interventions.plan must be a non-empty list")
    intervention_plan: list[dict[str, Any]] = []
    for index, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            raise ValueError("every interventions.plan item must be an object")
        intervention = dict(item)
        intervention["index"] = index
        intervention_plan.append(intervention)
    tasks: list[tuple[str, dict[str, str], str, dict[str, Any]]] = []

    with input_path.open("r", encoding="utf-8") as records:
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
                    counts["skipped_complex_script_assets"] += len(intervention_plan)
                    continue
                translations[language] = text
            if len(translations) < 2:
                counts["skipped_families"] += 1
                continue
            split = str(record.get("split") or _assign_split(base_family, config))
            if split == "dev":
                split = "validation"
            for intervention in intervention_plan:
                tasks.append((base_family, translations, split, intervention))

    generation = {
        name: int(config.generation[name])
        for name in (
            "seed",
            "width",
            "height",
            "min_font_size",
            "max_font_size",
            "curve_amplitude_min",
            "curve_amplitude_max",
            "margin",
            "stroke_width",
            "shadow_offset",
            "region_width",
            "region_padding",
            "image_safe_margin",
            "placement_attempts",
        )
    }
    context = {
        "output_dir": str(output_dir),
        "directories": {name: str(path) for name, path in directories.items()},
        "backgrounds": [str(path) for path in backgrounds],
        "font_paths": [str(path) for path in font_paths],
        "font_family": font_family,
        "generation": generation,
        "quality": {
            "quarantine_unexpected_errors": bool(config.quality.quarantine_unexpected_errors),
        },
    }
    configured_workers = int(config.generation.num_workers)
    cpu_limit = _cgroup_cpu_count()
    reserve_cpus = max(0, int(config.generation.reserve_cpus))
    workers = configured_workers if configured_workers > 0 else max(1, cpu_limit - reserve_cpus)
    workers = min(workers, cpu_limit)
    print(
        json.dumps(
            {
                "render_tasks": len(tasks),
                "render_workers": workers,
                "detected_cpu_limit": cpu_limit,
                "reserved_cpus": reserve_cpus,
            }
        )
    )
    rejection_log_interval = max(1, int(config.quality.log_every_rejections))
    processed_families = 0

    with (
        temporary.open("w", encoding="utf-8") as asset_stream,
        pairs_temporary.open("w", encoding="utf-8") as pair_stream,
        rejected_temporary.open("w", encoding="utf-8") as rejected_stream,
        ProcessPoolExecutor(
            max_workers=workers, initializer=_init_render_worker, initargs=(context,)
        ) as executor,
    ):
        for result in executor.map(_render_family_task, tasks, chunksize=1):
            processed_families += 1
            if result.skipped_languages:
                counts["skipped_missing_font_assets"] += len(result.skipped_languages)
                counts["families_with_partial_language_coverage"] += 1
            if result.rejection is not None:
                rejected_stream.write(json.dumps(result.rejection, ensure_ascii=False) + "\n")
                counts["rejected_families"] += 1
                counts[f"rejected/{result.rejection['reason']}"] += 1
                rejected = counts["rejected_families"]
                if rejected == 1 or rejected % rejection_log_interval == 0:
                    print(
                        f"WARNING: quarantined families={rejected}, "
                        f"latest={result.family_id}, reason={result.rejection['reason']}, "
                        f"processed={processed_families}/{len(tasks)}",
                        flush=True,
                    )
                continue
            family_assets = result.assets
            if len(family_assets) < 2:
                counts["skipped_render_families"] += 1
                continue
            for row in family_assets:
                asset_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["assets"] += 1
                counts[f"assets/{row['split']}"] += 1
            counts["pairs"] += _write_pairs(pair_stream, family_assets, config)
            counts["families"] += 1
            if processed_families % 100 == 0:
                print(
                    f"progress: processed={processed_families}/{len(tasks)}, "
                    f"accepted={counts['families']}, rejected={counts['rejected_families']}",
                    flush=True,
                )

    temporary.replace(manifest)
    pairs_temporary.replace(pairs_manifest)
    rejected_temporary.replace(rejected_manifest)
    counts["causal_pairs"] = _write_causal_pairs(manifest, causal_pairs_manifest)
    attempted = len(tasks)
    rejected = counts["rejected_families"]
    rejected_fraction = rejected / attempted if attempted else 0.0
    report = {
        **counts,
        "attempted_families": attempted,
        "accepted_families": counts["families"],
        "rejected_fraction": rejected_fraction,
        "manifest": str(manifest),
        "pairs_manifest": str(pairs_manifest),
        "causal_pairs_manifest": str(causal_pairs_manifest),
        "rejected_manifest": str(rejected_manifest),
    }
    quality_temporary = quality_report.with_suffix(quality_report.suffix + ".tmp")
    quality_temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    quality_temporary.replace(quality_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not counts["families"]:
        raise RuntimeError(f"No valid families were generated; inspect {rejected_manifest}")
    max_rejected = float(config.quality.max_rejected_fraction)
    if bool(config.quality.fail_on_excess_rejections) and rejected_fraction > max_rejected:
        raise RuntimeError(
            f"Rejected {rejected_fraction:.2%} of families, limit is {max_rejected:.2%}; "
            f"inspect {quality_report}"
        )
    return manifest


def validate_dataset(config: DictConfig) -> dict[str, int]:
    """Fail fast on leakage, broken files, dimensions, or invalid supervision masks."""
    output_dir = Path(config.output.dir)
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    causal_pairs_manifest = output_dir / str(config.output.causal_pairs_manifest)
    assets: dict[str, dict[str, Any]] = {}
    base_splits: dict[str, str] = {}
    family_languages: defaultdict[str, set[str]] = defaultdict(set)
    family_curves: dict[str, str] = {}
    counts: Counter[str] = Counter()

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

            images = {
                name: Image.open(output_dir / row[name])
                for name in ("image", "glyph_mask", "region_mask", "baseline_mask", "background")
            }
            expected = (row["width"], row["height"])
            if any(image.size != expected for image in images.values()):
                raise ValueError(f"dimension mismatch for {row['id']}")
            outside = ImageChops.subtract(
                images["glyph_mask"].convert("L"), images["region_mask"].convert("L")
            )
            if outside.getbbox() is not None:
                raise ValueError(f"glyph mask escapes region mask for {row['id']}")
            glyph_bbox = images["glyph_mask"].getbbox()
            if glyph_bbox is None or images["region_mask"].getbbox() is None:
                raise ValueError(f"empty mask for {row['id']}")
            safe_margin = int(config.generation.image_safe_margin)
            if (
                glyph_bbox[0] < safe_margin
                or glyph_bbox[1] < safe_margin
                or glyph_bbox[2] > expected[0] - safe_margin
                or glyph_bbox[3] > expected[1] - safe_margin
            ):
                raise ValueError(f"glyph mask touches the unsafe image boundary for {row['id']}")
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

    factor_fields = {
        "style": "style_id",
        "layout": "layout_id",
        "background": "background_id",
    }
    with causal_pairs_manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            pair = json.loads(line)
            source, target = assets.get(pair["source_id"]), assets.get(pair["target_id"])
            if source is None or target is None:
                raise ValueError(f"dangling causal pair at line {line_number}")
            if (
                source["base_family_id"] != target["base_family_id"]
                or source["language"] != target["language"]
                or source["text"] != target["text"]
                or source["split"] != target["split"]
            ):
                raise ValueError(f"invalid causal pair identity at line {line_number}")
            changed_factor = pair["changed_factor"]
            if changed_factor not in factor_fields:
                raise ValueError(f"unknown intervention factor at line {line_number}")
            changed = {factor for factor, field in factor_fields.items() if source[field] != target[field]}
            if changed != {changed_factor}:
                raise ValueError(
                    f"non-atomic intervention at line {line_number}: expected "
                    f"{changed_factor}, changed={sorted(changed)}"
                )
            counts["causal_pairs"] += 1

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


def _read_hf_credentials(config: DictConfig) -> dict[str, Any]:
    credentials_path = Path(str(config.hf.credentials_file)).expanduser()
    if not credentials_path.is_file():
        return {}
    try:
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {credentials_path}: {error}") from error
    if not isinstance(credentials, dict):
        raise ValueError(f"{credentials_path} must contain a JSON object")
    return credentials


def _resolve_hf_token(config: DictConfig) -> str | None:
    explicit = config.hf.token
    if explicit:
        return str(explicit).strip()
    credentials_path = Path(str(config.hf.credentials_file)).expanduser()
    credentials = _read_hf_credentials(config)
    if credentials:
        token_key = str(config.hf.token_key)
        token = credentials.get(token_key)
        if not isinstance(token, str) or not token.strip():
            raise ValueError(f"{credentials_path} must contain a non-empty string field {token_key!r}")
        return token.strip()
    environment_token = os.environ.get("HF_TOKEN")
    return environment_token.strip() if environment_token else None


def _resolve_hf_repo(config: DictConfig) -> str | None:
    if config.hf.repo:
        return str(config.hf.repo).strip()
    credentials = _read_hf_credentials(config)
    repo_key = str(config.hf.repo_key)
    repo = credentials.get(repo_key)
    if repo is not None:
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError(f"hf credential field {repo_key!r} must be a non-empty string")
        return repo.strip()
    environment_repo = os.environ.get("HF_DATASET_REPO")
    return environment_repo.strip() if environment_repo else None


def push_to_hub(config: DictConfig) -> None:
    """Publish relational assets/pairs configs without duplicating image bytes in pair rows."""
    from datasets import Dataset, DatasetDict, Features, Value
    from datasets import Image as HFImage
    from huggingface_hub import HfApi

    repo_id = _resolve_hf_repo(config)
    token = _resolve_hf_token(config)
    if not repo_id:
        raise ValueError("Set hf.repo=owner/dataset or HF_DATASET_REPO")
    if not token:
        raise ValueError(f"Put hf_token in {config.hf.credentials_file}, set hf.token, or export HF_TOKEN")
    output_dir = Path(config.output.dir).resolve()
    manifest = output_dir / str(config.output.manifest)
    pairs_manifest = output_dir / str(config.output.pairs_manifest)
    causal_pairs_manifest = output_dir / str(config.output.causal_pairs_manifest)
    image = HFImage()
    asset_features = Features(
        {
            "id": Value("string"),
            "base_family_id": Value("string"),
            "family_id": Value("string"),
            "variant": Value("int32"),
            "intervention_name": Value("string"),
            "intervention_type": Value("string"),
            "changed_factor": Value("string"),
            "style_id": Value("string"),
            "layout_id": Value("string"),
            "background_id": Value("string"),
            "font_family": Value("string"),
            "style_level": Value("int32"),
            "layout_level": Value("int32"),
            "background_level": Value("int32"),
            "split": Value("string"),
            "language": Value("string"),
            "text": Value("string"),
            "image": image,
            "glyph_mask": image,
            "region_mask": image,
            "baseline_mask": image,
            "background": image,
            "font": Value("string"),
            "font_size": Value("int32"),
            "placement_attempt": Value("int32"),
            "curve": Value("string"),
            "geometry_seed": Value("int64"),
            "width": Value("int32"),
            "height": Value("int32"),
        }
    )
    translation_pair_features = Features(
        {
            name: Value("string")
            for name in (
                "id",
                "pair_type",
                "intervention_type",
                "changed_factor",
                "base_family_id",
                "family_id",
                "split",
                "source_id",
                "target_id",
                "source_language",
                "target_language",
                "source_text",
                "target_text",
            )
        }
    )
    causal_pair_features = Features(
        {
            name: Value("string")
            for name in (
                "id",
                "pair_type",
                "base_family_id",
                "split",
                "language",
                "text",
                "source_id",
                "target_id",
                "intervention_type",
                "changed_factor",
                "source_style_id",
                "target_style_id",
                "source_layout_id",
                "target_layout_id",
                "source_background_id",
                "target_background_id",
            )
        }
    )
    asset_splits = _available_splits(manifest)
    pair_splits = _available_splits(pairs_manifest)
    assets = DatasetDict(
        {
            split: Dataset.from_generator(
                _asset_generator,
                gen_kwargs={"manifest": str(manifest), "output_dir": str(output_dir), "split": split},
                features=asset_features,
            )
            for split in asset_splits
        }
    )
    pairs = DatasetDict(
        {
            split: Dataset.from_generator(
                _pair_generator,
                gen_kwargs={"manifest": str(pairs_manifest), "split": split},
                features=translation_pair_features,
            )
            for split in pair_splits
        }
    )
    causal_pairs = DatasetDict(
        {
            split: Dataset.from_generator(
                _pair_generator,
                gen_kwargs={"manifest": str(causal_pairs_manifest), "split": split},
                features=causal_pair_features,
            )
            for split in _available_splits(causal_pairs_manifest)
        }
    )
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=bool(config.hf.private), exist_ok=True)
    assets.push_to_hub(
        repo_id,
        config_name="assets",
        token=token,
        private=bool(config.hf.private),
        max_shard_size=str(config.hf.max_shard_size),
    )
    pairs.push_to_hub(
        repo_id,
        config_name="pairs",
        token=token,
        private=bool(config.hf.private),
        max_shard_size=str(config.hf.max_shard_size),
    )
    causal_pairs.push_to_hub(
        repo_id,
        config_name="causal_pairs",
        token=token,
        private=bool(config.hf.private),
        max_shard_size=str(config.hf.max_shard_size),
    )
    print(f"Published https://huggingface.co/datasets/{repo_id} (configs: assets, pairs, causal_pairs)")
