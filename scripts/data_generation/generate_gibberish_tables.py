#!/usr/bin/env python3
"""Synthetic Gibberish English Table Generator using PubTabNet Seed Tables.

Generates two synthetic table datasets:
1. Random Cells Gibberish: Random cells (40%-60%) replaced with gibberish English text.
2. All Cells Gibberish: 100% of cell contents replaced with gibberish English text.

Key Features:
- Strict word length capping: Every generated gibberish word is capped to max 30 characters.
- Varied English gibberish: Syllabic pronounceable words, random letter strings,
  alphanumeric codes, and numeric-like table values.
- PubTabNet evaluation split ("val" / "eval") with seed 2026.
- Stratified sampling: 30% Simple, 30% Medium, 40% Complex.
- MUSTARD-style layout stressors (borderless_dense, underlined_header, inverse_banner) on 35% of tables.
- Printed document degradation (ink bleed, streaks, toner speckles, smudging, soft blur, JPEG compression).
- High performance multi-worker Playwright rendering.
- Outputs TSR bounding boxes, structure tokens, cell annotations, and preview HTML.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent OpenMP / BLAS thread contention across multiprocessing workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONNOUSERSITE"] = "1"
sys.path = [p for p in sys.path if ".local/lib" not in p]

import warnings
from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

import argparse
import asyncio
import copy
import html as html_lib
import io
import json
import logging
import multiprocessing as mp
import random
import string
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from bs4 import BeautifulSoup, NavigableString
from PIL import Image as PILImage, ImageDraw

LOG = logging.getLogger("gibberish_tables")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = Path.home() / "scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/PubTabNet_2.0.0.jsonl"
DEFAULT_CACHE_DIR = SCRIPT_DIR

STYLE_CSS = {
    "booktabs": "table { border-collapse: collapse; width: max-content; height: auto; border-top: 1.6px solid #222; border-bottom: 1.6px solid #222; } td, th { border: none; padding: 6px 8px; text-align: left; vertical-align: middle; } thead > tr:last-child > td, thead > tr:last-child > th { border-bottom: 1.1px solid #222; }",
    "horizontal_rules": "table { border-collapse: collapse; width: max-content; height: auto; border-top: 1px solid #555; border-bottom: 1.2px solid #555; } td, th { border: none; border-bottom: 0.65px solid #c4c4c4; padding: 6px 8px; text-align: left; vertical-align: middle; } thead > tr:last-child > td, thead > tr:last-child > th { border-bottom: 1.4px solid #555; }",
    "basic_white": "table { border-collapse: separate; width: max-content; height: auto; border-spacing: 0; border: none; background: #fff; } td, th { border: none !important; background: #fff; padding: 6px 8px; text-align: left; vertical-align: middle; }",
    "no_rules": "table { border-collapse: separate; width: max-content; height: auto; border-spacing: 0; border: none; } td, th { border: none !important; padding: 6px 8px; text-align: left; vertical-align: middle; }",
    "vertical_rules": "table { border-collapse: collapse; width: max-content; height: auto; border: none; } td, th { border-top: none !important; border-bottom: none !important; border-left: 0.7px solid #c8c8c8; border-right: 0.7px solid #c8c8c8; padding: 6px 8px; text-align: left; vertical-align: middle; }",
    "full_grid": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: 1px solid #777; padding: 6px 7px; text-align: left; vertical-align: middle; } thead > tr > td, thead > tr > th { font-weight: 600; }",
    "minimal": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: none; border-bottom: 0.55px solid #dedede; padding: 8px 10px; text-align: left; vertical-align: middle; } thead > tr:last-child > td, thead > tr:last-child > th { border-bottom: 1px solid #999; }",
    "financial": "table { border-collapse: collapse; width: max-content; height: auto; border-top: 1.2px solid #444; } td, th { border: none; border-bottom: 0.65px solid #c9cdd2; padding: 6px 7px; text-align: left; vertical-align: middle; } thead > tr:last-child > td, thead > tr:last-child > th { border-bottom: 1.4px solid #444; }",
    "dense": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: 0.8px solid #777; padding: 3px 4px; line-height: 1.15; text-align: left; vertical-align: middle; }",
    # MUSTARD-inspired stressor styles:
    "borderless_dense": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: none !important; padding: 2px 5px; text-align: left; vertical-align: top; line-height: 1.12; }",
    "underlined_header": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: none !important; padding: 3px 6px; text-align: left; vertical-align: top; } thead > tr > th, thead > tr > td, tr:first-child > th, tr:first-child > td { text-decoration: underline; text-underline-offset: 2px; font-weight: 700; }",
    "inverse_banner": "table { border-collapse: collapse; width: max-content; height: auto; } td, th { border: 1px solid #222; padding: 5px 7px; text-align: left; vertical-align: middle; } thead > tr:first-child > th, thead > tr:first-child > td, tr:first-child > th { background: #111 !important; color: #fff !important; font-weight: 700; }",
}
STYLE_NAMES = list(STYLE_CSS.keys())

BACKGROUND_PALETTES = {
    "none": "",
    "pale_blue": "table { background: #ffffff; } thead > tr > td, thead > tr > th { background: #e4edf9 !important; } tbody > tr:nth-child(even) > td, tbody > tr:nth-child(even) > th { background: #f3f7fc !important; }",
    "pale_green": "table { background: #ffffff; } thead > tr > td, thead > tr > th { background: #e6f1e9 !important; } tbody > tr:nth-child(even) > td, tbody > tr:nth-child(even) > th { background: #f4f9f5 !important; }",
}
COLOR_PALETTE_NAMES = ["pale_blue", "pale_green"]
COLOR_FILL_PROBABILITY = 0.20

FONTS = ["Liberation Serif", "FreeSerif", "DejaVu Serif", "Liberation Sans", "DejaVu Sans"]


def setup_logging(prefix: str = "GIBBERISH_TSR") -> None:
    fmt = f"%(asctime)s | {prefix} | %(levelname)s | %(message)s" if prefix else "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S", force=True)


def apply_printed_doc_degradation(img: PILImage.Image, rng: random.Random) -> PILImage.Image:
    """Applies realistic printed document artifacts: ink bleed, streaks, speckles, smudge, fade, blur, JPEG noise."""
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    cv_img = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    h, w = cv_img.shape[:2]

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    if rng.random() < 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)
        bleed_areas = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(text_mask))
        bleed_intensity = rng.uniform(0.1, 0.35)
        cv_img[bleed_areas > 0] = (cv_img[bleed_areas > 0] * (1.0 - bleed_intensity)).astype(np.uint8)

    if rng.random() < 0.45:
        num_streaks = rng.randint(2, 5)
        for _ in range(num_streaks):
            y = rng.randint(0, max(0, h - 1))
            thickness = rng.randint(1, 3)
            streak_mask = np.zeros_like(gray)
            cv2.line(streak_mask, (0, y), (w, y), 255, thickness)
            target = cv2.bitwise_and(text_mask, streak_mask)
            cv_img[target > 0] = np.clip(cv_img[target > 0] * 1.5, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        noise = np_rng.random((h, w))
        speckle_zone = cv2.dilate(text_mask, np.ones((15, 15), np.uint8))
        speckles = (noise < 0.0018).astype(np.uint8) * 255
        target_speckles = cv2.bitwise_and(speckles, speckle_zone)
        cv_img[target_speckles > 0] = (cv_img[target_speckles > 0] * rng.uniform(0.15, 0.45)).astype(np.uint8)

    if rng.random() < 0.35:
        kernel_size = rng.choice([3, 5])
        motion_blur_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        motion_blur_kernel[kernel_size // 2, :] = 1.0 / kernel_size
        smudged = cv2.filter2D(cv_img, -1, motion_blur_kernel)
        smear_zone = cv2.dilate(text_mask, np.ones((5, 5), np.uint8))
        blend_mask = (smear_zone.astype(np.float32) / 255.0)[..., None]
        cv_img = (cv_img * (1.0 - blend_mask) + smudged * blend_mask).astype(np.uint8)

    if rng.random() < 0.45 and h >= 50 and w >= 50:
        fading_map = np_rng.random((max(2, h // 50), max(2, w // 50))).astype(np.float32)
        fading_map = cv2.resize(fading_map, (w, h), interpolation=cv2.INTER_CUBIC)
        fade_mask = (fading_map > 0.62).astype(np.uint8) * 255
        target_fade = cv2.bitwise_and(text_mask, fade_mask)
        cv_img[target_fade > 0] = np.clip(cv_img[target_fade > 0] * 1.45, 0, 255).astype(np.uint8)

    blur_sigma = rng.uniform(0.6, 1.15)
    cv_img = cv2.GaussianBlur(cv_img, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb_img)

    buffer = io.BytesIO()
    jpeg_quality = rng.randint(25, 45)
    pil_img.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    return PILImage.open(buffer).convert("RGB")


# ==============================================================================
# GIBBERISH ENGLISH TEXT GENERATOR (Strict <= 30 chars per word)
# ==============================================================================

class GibberishGenerator:
    """Generates random English gibberish text with strict capping to max 30 characters per word."""

    def __init__(self, max_word_len: int = 30):
        self.max_word_len = max(1, min(30, max_word_len))
        self.vowels = "aeiouy"
        self.consonants = "bcdfghjklmnpqrstvwxz"
        self.letters = string.ascii_lowercase
        self.digits = string.digits

    def random_word(self, rng: random.Random) -> str:
        """Generates a single gibberish word of length <= max_word_len."""
        style = rng.random()

        # Target word length distribution: mostly realistic 2-12 chars, occasionally up to 30
        len_mode = rng.choices(["short", "medium", "longer", "max_length"], weights=[0.45, 0.35, 0.15, 0.05])[0]
        if len_mode == "short":
            target_len = rng.randint(2, 6)
        elif len_mode == "medium":
            target_len = rng.randint(6, 12)
        elif len_mode == "longer":
            target_len = rng.randint(12, 20)
        else:
            target_len = rng.randint(20, self.max_word_len)

        target_len = min(target_len, self.max_word_len)

        if style < 0.45:
            # 1. Syllable-based pronounceable pseudo-word (e.g. "kestram", "zorpux", "flonvida")
            chars = []
            use_vowel = rng.choice([True, False])
            while len(chars) < target_len:
                c = rng.choice(self.vowels) if use_vowel else rng.choice(self.consonants)
                chars.append(c)
                use_vowel = not use_vowel
            word = "".join(chars[:target_len])

        elif style < 0.70:
            # 2. Random ASCII lowercase string (e.g. "xqrfvb", "plokmn")
            word = "".join(rng.choice(self.letters) for _ in range(target_len))

        elif style < 0.85:
            # 3. Alphanumeric / code-like table entry (e.g. "a4-9x", "kp82", "z091b")
            chars = self.letters + self.digits
            word_chars = [rng.choice(chars) for _ in range(target_len)]
            if target_len >= 5 and rng.random() < 0.3:
                hyphen_pos = rng.randint(2, target_len - 2)
                word_chars[hyphen_pos] = "-"
            word = "".join(word_chars)

        else:
            # 4. Numeric-like table entry (e.g. "14.28", "982", "54.1%")
            val = round(rng.uniform(0.1, 9999.0), rng.randint(0, 2))
            word = str(val) if val != int(val) else str(int(val))
            if rng.random() < 0.25:
                word += "%"

        # Apply capitalization variations
        case_roll = rng.random()
        if case_roll < 0.25:
            word = word.capitalize()
        elif case_roll < 0.35:
            word = word.upper()

        # Strict cap
        word = word[:self.max_word_len]
        return word

    def generate_cell_text(self, rng: random.Random, num_words: Optional[int] = None) -> str:
        """Generates space-separated gibberish words, strictly validating each word <= max_word_len."""
        if num_words is None:
            # Tables typically contain 1 to 4 words per cell
            num_words = rng.choices([1, 2, 3, 4, 5], weights=[0.45, 0.30, 0.15, 0.07, 0.03])[0]

        words = [self.random_word(rng) for _ in range(num_words)]
        # Filter and enforce cap
        valid_words = [w[:self.max_word_len] for w in words if w]
        if not valid_words:
            valid_words = [self.random_word(rng)[:self.max_word_len]]

        # Extra verification
        for w in valid_words:
            assert len(w) <= self.max_word_len, f"Word '{w}' exceeds {self.max_word_len} chars!"

        return " ".join(valid_words)


# ==============================================================================
# SEED TABLE EXTRACTION (Filtered for Eval Split + Stratified Logic)
# ==============================================================================

def source_cells(record: dict) -> list:
    h = record.get("html", {}) or {}
    return h.get("cells") or h.get("cell") or []

def get_structure_tokens(record: dict) -> list:
    h = record.get("html", {}) or {}
    structure = h.get("structure", {}) or {}
    return structure.get("tokens") or []

def source_cell_text(cell: dict) -> str:
    tokens = cell.get("tokens") or []
    raw = "".join(str(t) for t in tokens)
    return unicodedata.normalize("NFC", html_lib.unescape(BeautifulSoup(raw, "html.parser").get_text(separator="", strip=False))).strip()

def build_table_soup(structure_tokens: Sequence[str]) -> Tuple[BeautifulSoup, Any]:
    structure_html = "".join(str(t) for t in structure_tokens)
    soup = BeautifulSoup(structure_html, "html.parser")
    table = soup.find("table")
    if table is None:
        soup = BeautifulSoup(f"<table>{structure_html}</table>", "html.parser")
        table = soup.find("table")
    return soup, table

def inspect_table(record: dict):
    tokens = get_structure_tokens(record)
    annotations = source_cells(record)
    if not tokens:
        raise ValueError("missing tokens")
    soup, table = build_table_soup(tokens)
    dom_cells = [c for c in table.find_all(["td", "th"]) if c.find_parent("table") is table]
    if len(dom_cells) != len(annotations):
        raise ValueError("mismatch")

    rows = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    occupied = set()
    cell_layout = []
    max_row, max_col = 0, 0
    row_lengths = []

    for r_idx, tr in enumerate(rows):
        direct_cells = [c for c in tr.find_all(["td", "th"], recursive=False) if c.find_parent("table") is table]
        row_lengths.append(len(direct_cells))
        c_idx = 0
        for cell in direct_cells:
            while any((rr, cc) in occupied for rr in range(r_idx, r_idx + 1) for cc in range(c_idx, c_idx + 1)):
                c_idx += 1
            rowspan = int(cell.get("rowspan", 1)) if str(cell.get("rowspan", 1)).isdigit() else 1
            colspan = int(cell.get("colspan", 1)) if str(cell.get("colspan", 1)).isdigit() else 1
            for rr in range(r_idx, r_idx + rowspan):
                for cc in range(c_idx, c_idx + colspan):
                    occupied.add((rr, cc))
            cell_layout.append({
                "row_start": r_idx, "row_end": r_idx + rowspan - 1,
                "col_start": c_idx, "col_end": c_idx + colspan - 1,
                "rowspan": rowspan, "colspan": colspan, "tag": cell.name
            })
            max_row = max(max_row, r_idx + rowspan)
            max_col = max(max_col, c_idx + colspan)
            c_idx += colspan

    metrics = {
        "rows": max(max_row, len(rows)), "cols": max_col, "cells": len(dom_cells),
        "span_cells": len([c for c in cell_layout if c["rowspan"] > 1 or c["colspan"] > 1]),
        "row_length_variation": len(set(row_lengths)) > 1,
        "row_cell_counts": row_lengths,
        "span_ratio": round(len([c for c in cell_layout if c["rowspan"] > 1 or c["colspan"] > 1]) / max(1, len(dom_cells)), 6),
    }
    return metrics, soup, table, dom_cells, cell_layout

def complexity_from_metrics(metrics: dict) -> str:
    if metrics["rows"] >= 30 or metrics["cols"] >= 10 or metrics["cells"] >= 150 or metrics["span_cells"] >= 6:
        return "complex"
    if metrics["rows"] >= 12 or metrics["cols"] >= 6 or metrics["cells"] >= 50 or metrics["span_cells"] >= 1 or metrics["row_length_variation"]:
        return "medium"
    return "simple"

def generate_global_pools(
    input_jsonl: Path,
    num_tables: int,
    seed: int,
    split: str = "val",
    cache_file: Optional[Path] = None,
    is_sanity_check: bool = False
) -> list:
    """Samples exact 30% Simple / 30% Medium / 40% Complex master tables from PubTabNet eval split."""
    possible_caches = []
    if cache_file:
        possible_caches.append(cache_file)
    possible_caches.extend([
        DEFAULT_CACHE_DIR / f"seed_master_pool_{split}_{num_tables}_seed{seed}.json",
        DEFAULT_CACHE_DIR / f"seed_master_pool_{num_tables}_seed{seed}.json",
        DEFAULT_CACHE_DIR / f"seed_master_pool_500_seed{seed}.json",
    ])

    for cpath in possible_caches:
        if cpath.exists():
            LOG.info(f"Loading cached master pool from: {cpath}")
            try:
                with cpath.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if is_sanity_check:
                    return cached[:num_tables]
                if len(cached) == num_tables:
                    return cached
                if len(cached) > num_tables:
                    return cached[:num_tables]
                LOG.info(f"Cached pool size ({len(cached)}) < requested ({num_tables}), checking next...")
            except Exception as e:
                LOG.warning(f"Failed to read cache file {cpath}: {e}")

    rng = random.Random(seed)
    n_simple = int(num_tables * 0.3)
    n_medium = int(num_tables * 0.3)
    n_complex = num_tables - n_simple - n_medium

    caps = {"simple": n_simple, "medium": n_medium, "complex": n_complex}
    pools = {"simple": [], "medium": [], "complex": []}
    seen_ids = set()
    total_scanned = 0

    valid_splits = {split.lower(), "val", "eval"} if split in ["val", "eval"] else {split.lower()}
    LOG.info(f"Scanning {input_jsonl} (split: {valid_splits}) for {caps['simple']} Simple / {caps['medium']} Medium / {caps['complex']} Complex...")

    with input_jsonl.open("r", encoding="utf-8") as f:
        while True:
            lines = f.readlines(1000000)
            if not lines:
                break
            for line in lines:
                if not line.strip():
                    continue
                total_scanned += 1
                try:
                    record = json.loads(line)
                    rec_split = str(record.get("split", "")).lower()
                    if valid_splits and rec_split not in valid_splits:
                        continue

                    sid = f"{record.get('split', '')}/{record.get('filename', '')}"
                    if sid in seen_ids:
                        continue

                    metrics, _, _, _, layout = inspect_table(record)
                    if metrics["rows"] < 2:
                        continue
                    bucket = complexity_from_metrics(metrics)

                    if len(pools[bucket]) < caps[bucket]:
                        seen_ids.add(sid)
                        pools[bucket].append({
                            "record": record,
                            "source_id": sid,
                            "complexity_bucket": bucket,
                            "complexity_metrics": metrics,
                            "cell_layout": layout
                        })
                except Exception:
                    continue

            if all(len(pools[k]) >= caps[k] for k in caps):
                LOG.info(f"Target reached! Filled all complexity buckets after scanning {total_scanned:,} records.")
                break

    master_list = pools["simple"] + pools["medium"] + pools["complex"]
    rng.shuffle(master_list)

    if cache_file:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(master_list, f)
            LOG.info(f"Cached master pool of {len(master_list)} tables to {cache_file}")
        except Exception as e:
            LOG.warning(f"Could not write cache file: {e}")

    return master_list


def build_translated_soup(record: dict, translated_texts: Sequence[str]) -> Tuple[BeautifulSoup, Any, list, list]:
    tokens = get_structure_tokens(record)
    soup, table = build_table_soup(tokens)
    dom_cells = [cell for cell in table.find_all(["td", "th"]) if cell.find_parent("table") is table]
    annotations = source_cells(record)

    for dom_cell, ann, text in zip(dom_cells, annotations, translated_texts):
        dom_cell.clear()
        tokens_ann = [str(t).lower() for t in (ann.get("tokens") or [])]
        bold, italic = "<b>" in tokens_ann, "<i>" in tokens_ann
        node = NavigableString(text) if text else NavigableString("")
        if bold and italic:
            strong, em = soup.new_tag("strong"), soup.new_tag("em")
            em.append(node)
            strong.append(em)
            dom_cell.append(strong)
        elif bold:
            strong = soup.new_tag("strong")
            strong.append(node)
            dom_cell.append(strong)
        elif italic:
            em = soup.new_tag("em")
            em.append(node)
            dom_cell.append(em)
        else:
            dom_cell.append(node)
    return soup, table, dom_cells, annotations


# ==============================================================================
# CELL LAYOUT & STRUCTURE TOKEN EXTRACTION
# ==============================================================================

def extract_cell_layout(table_node) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Calculates table layout, row/col spans, and finds header vs body cells."""
    rows = [tr for tr in table_node.find_all("tr") if tr.find_parent("table") is table_node]
    occupied = set()
    cell_layout = []
    max_row, max_col = 0, 0
    row_lengths = []

    for r_idx, tr in enumerate(rows):
        direct_cells = [c for c in tr.find_all(["td", "th"], recursive=False) if c.find_parent("table") is table_node]
        row_lengths.append(len(direct_cells))
        c_idx = 0
        for cell in direct_cells:
            while (r_idx, c_idx) in occupied:
                c_idx += 1
            rowspan = int(cell.get("rowspan", 1)) if str(cell.get("rowspan", 1)).isdigit() else 1
            colspan = int(cell.get("colspan", 1)) if str(cell.get("colspan", 1)).isdigit() else 1
            for rr in range(r_idx, r_idx + rowspan):
                for cc in range(c_idx, c_idx + colspan):
                    occupied.add((rr, cc))
            is_in_thead = bool(cell.find_parent("thead"))
            cell_layout.append({
                "cell_node": cell,
                "row_start": r_idx,
                "row_end": r_idx + rowspan - 1,
                "col_start": c_idx,
                "col_end": c_idx + colspan - 1,
                "rowspan": rowspan,
                "colspan": colspan,
                "tag": cell.name,
                "is_header": (cell.name == "th" or is_in_thead),
                "original_text": unicodedata.normalize("NFC", html_lib.unescape(cell.get_text(separator="", strip=True))).strip()
            })
            max_row = max(max_row, r_idx + rowspan)
            max_col = max(max_col, c_idx + colspan)
            c_idx += colspan

    metrics = {
        "rows": max(max_row, len(rows)),
        "cols": max_col,
        "cells": len(cell_layout),
        "span_cells": len([c for c in cell_layout if c["rowspan"] > 1 or c["colspan"] > 1]),
        "row_length_variation": len(set(row_lengths)) > 1
    }
    return metrics, cell_layout


def extract_structure_tokens_from_table(table_node) -> List[str]:
    """Generates standard PubTabNet structure tokens from the table DOM."""
    tokens = ["<html>", "<body>", "<table>"]

    def traverse(node):
        if node.name in ["thead", "tbody"]:
            tokens.append(f"<{node.name}>")
            for child in node.children:
                if child.name:
                    traverse(child)
            tokens.append(f"</{node.name}>")
        elif node.name == "tr":
            tokens.append("<tr>")
            for child in node.children:
                if child.name:
                    traverse(child)
            tokens.append("</tr>")
        elif node.name in ["td", "th"]:
            attrs = []
            if node.get("rowspan") and str(node.get("rowspan")).isdigit() and int(node.get("rowspan")) > 1:
                attrs.append(f'rowspan="{node.get("rowspan")}"')
            if node.get("colspan") and str(node.get("colspan")).isdigit() and int(node.get("colspan")) > 1:
                attrs.append(f'colspan="{node.get("colspan")}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            tokens.append(f"<{node.name}{attr_str}>")
            tokens.append(f"</{node.name}>")

    for child in table_node.children:
        if child.name:
            traverse(child)

    tokens.append("</table>")
    tokens.append("</body>")
    tokens.append("</html>")
    return tokens


def make_html_document(table_markup: str, render_cfg: dict) -> str:
    family = render_cfg["font_family"].replace("\\", "\\\\").replace('"', '\\"')
    css = STYLE_CSS[render_cfg["style_family"]]
    bg_css = BACKGROUND_PALETTES.get(render_cfg.get("background_variant", "none"), "")
    header_css = "table > thead > tr > th, table > thead > tr > td { font-weight: 700 !important; }" if render_cfg.get("bold_header") else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; width: max-content; height: max-content; background: #fff; color: #111; }}
body {{ font-family: "{family}", "FreeSerif", serif; font-size: {render_cfg['font_size_px']}px; font-weight: {render_cfg['font_weight']}; line-height: {render_cfg['line_height']}; }}
.table-frame {{ padding: 16px; background: #fff; width: max-content; height: max-content; }}
table {{ table-layout: auto; width: auto; max-width: none; margin: 0; font-variant-numeric: tabular-nums; }}
td, th {{ vertical-align: middle; overflow-wrap: break-word; word-break: normal; padding: {render_cfg['cell_padding_px']}px; }}
.gibberish-text {{
    display: inline-block;
    word-break: normal;
    overflow-wrap: break-word;
    font-family: inherit;
    font-size: inherit;
    line-height: inherit;
}}
{css}
{bg_css}
{header_css}
</style>
</head>
<body><div class="table-frame">{table_markup}</div></body>
</html>"""


async def measure_and_screenshot(page, out_png: Path, rng: random.Random, apply_degradation: bool = False) -> dict:
    table_locator = page.locator("table").first
    await table_locator.wait_for(state="visible", timeout=15000)
    await page.evaluate("() => document.fonts.ready")

    highres_path = out_png.with_name(f"{out_png.stem}.tmp_render{out_png.suffix}")
    await table_locator.screenshot(path=str(highres_path), animations="disabled", caret="hide", scale="device")

    with PILImage.open(highres_path) as highres:
        rgb_img = highres.convert("RGB")
        if apply_degradation:
            rgb_img = apply_printed_doc_degradation(rgb_img, rng)
        rgb_img.save(out_png, format="PNG", optimize=True)
    highres_path.unlink(missing_ok=True)

    layout = await page.evaluate("""() => {
      const table = document.querySelector('table');
      const tr = table.getBoundingClientRect();
      const cells = Array.from(table.querySelectorAll('td, th')).filter(el => el.closest('table') === table);
      const box = r => ({x: r.left - tr.left, y: r.top - tr.top, width: r.width, height: r.height});
      return {
        table: {width: tr.width, height: tr.height},
        cells: cells.map(el => ({
          ...box(el.getBoundingClientRect()),
          tag: el.tagName.toLowerCase(),
          rowspan: Number(el.getAttribute('rowspan') || 1),
          colspan: Number(el.getAttribute('colspan') || 1)
        }))
      };
    }""")

    tbox = layout["table"]
    with PILImage.open(out_png) as check_img:
        width_px, height_px = check_img.size

    sx = width_px / max(float(tbox["width"]), 1e-6)
    sy = height_px / max(float(tbox["height"]), 1e-6)

    def to_pixels(box: dict) -> dict:
        x0 = int(round(float(box["x"]) * sx))
        y0 = int(round(float(box["y"]) * sy))
        x1 = int(round((float(box["x"]) + float(box["width"])) * sx))
        y1 = int(round((float(box["y"]) + float(box["height"])) * sy))
        return {"xyxy": [x0, y0, x1, y1], "xywh": [x0, y0, x1 - x0, y1 - y0]}

    return {
        "image_width_px": width_px,
        "image_height_px": height_px,
        "cell_boxes_px": [to_pixels(b) for b in layout["cells"]]
    }


def draw_debug_bboxes(image_path: Path, cells_gt: list, out_debug_path: Path):
    """Draws color-coded bounding boxes on the table image for visual sanity checking."""
    with PILImage.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        for cell in cells_gt:
            x0, y0, x1, y1 = cell["rendered_bbox_xyxy_px"]
            if cell.get("is_gibberish"):
                # Gibberish cells in Crimson / Red
                draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 60), width=2)
            elif cell["is_header"]:
                # Header cells in Dodger Blue
                draw.rectangle([x0, y0, x1, y1], outline=(30, 144, 255), width=2)
            else:
                # Regular cells in Green
                draw.rectangle([x0, y0, x1, y1], outline=(34, 139, 34), width=1)
        img.save(out_debug_path, format="PNG")


# ==============================================================================
# ASYNC RENDERING WORKER (Sharded for fast parallel execution)
# ==============================================================================

async def async_render_shard(
    shard_id: int,
    indexed_samples: List[Tuple[int, dict]],
    shared_configs: List[dict],
    output_dir: Path,
    out_ann_path: Path,
    seed: int,
    mode: str,  # "random" or "all"
    min_gibberish_prob: float,
    max_gibberish_prob: float,
    fixed_gibberish_prob: Optional[float],
    max_gibberish_cells_per_table: int,
    max_word_len: int,
    apply_degradation: bool,
    draw_debug: bool
):
    from playwright.async_api import async_playwright

    logger = logging.getLogger(f"GIBBERISH_SHARD_{shard_id}")
    images_dir = output_dir / "images"
    debug_dir = output_dir / "images_debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    if draw_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)

    gibberish_gen = GibberishGenerator(max_word_len=max_word_len)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 2400, "height": 2400}, device_scale_factor=3)

        with out_ann_path.open("w", encoding="utf-8") as ann_f:
            for step, (idx, sample) in enumerate(indexed_samples):
                record = sample["record"]
                source_id = sample["source_id"]
                image_id = f"gibberish_table_{idx:05d}"
                out_png = images_dir / f"{image_id}.png"
                out_debug_png = debug_dir / f"{image_id}_debug.png"

                table_rng = random.Random(seed + idx)

                # 1. Build initial soup from PubTabNet text
                src_texts = [source_cell_text(c) for c in source_cells(record)]
                soup, table_node, _, _ = build_translated_soup(record, src_texts)

                # 2. Extract cell layout & determine headers vs body
                metrics, cell_layout = extract_cell_layout(table_node)
                total_cells = metrics["cells"]

                # 3. Determine Gibberish Replacement Indices
                if mode == "all":
                    # Replace 100% of cells
                    chosen_indices = set(range(total_cells))
                    gibberish_prob = 1.0
                else:
                    # Replace random cells (default 40%-60% or fixed)
                    if fixed_gibberish_prob is not None:
                        gibberish_prob = fixed_gibberish_prob
                    else:
                        gibberish_prob = table_rng.uniform(min_gibberish_prob, max_gibberish_prob)

                    num_to_inject = max(1, int(round(total_cells * gibberish_prob)))
                    if max_gibberish_cells_per_table and max_gibberish_cells_per_table > 0:
                        num_to_inject = min(num_to_inject, max_gibberish_cells_per_table)

                    body_indices = [i for i, c in enumerate(cell_layout) if not c["is_header"]]
                    if len(body_indices) >= num_to_inject:
                        chosen_indices = set(table_rng.sample(body_indices, num_to_inject))
                    else:
                        remaining = num_to_inject - len(body_indices)
                        other_indices = [i for i in range(total_cells) if i not in set(body_indices)]
                        chosen_indices = set(body_indices) | set(table_rng.sample(other_indices, min(remaining, len(other_indices))))

                # 4. Inject Gibberish English text into chosen cells
                cell_annotations = []
                for i, cell_info in enumerate(cell_layout):
                    dom_cell = cell_info["cell_node"]
                    if i in chosen_indices:
                        orig_text = cell_info["original_text"]
                        orig_words = len(orig_text.split()) if orig_text else 1
                        num_words = max(1, min(5, orig_words))
                        gibberish_text = gibberish_gen.generate_cell_text(table_rng, num_words=num_words)

                        # Verify strict <= 30 chars per word
                        for w in gibberish_text.split():
                            assert len(w) <= max_word_len, f"Word '{w}' in cell {i} exceeds {max_word_len} chars!"

                        dom_cell.clear()
                        span_tag = soup.new_tag("span")
                        span_tag["class"] = "gibberish-text"
                        span_tag.string = gibberish_text
                        dom_cell.append(span_tag)

                        cell_annotations.append({
                            "is_gibberish": True,
                            "text": gibberish_text,
                            "tokens": [gibberish_text]
                        })
                    else:
                        cell_annotations.append({
                            "is_gibberish": False,
                            "text": cell_info["original_text"],
                            "tokens": [cell_info["original_text"]] if cell_info["original_text"] else []
                        })

                # 5. Extract structure tokens for TSR
                structure_tokens = extract_structure_tokens_from_table(table_node)

                # 6. Render style configuration
                render_cfg = shared_configs[idx]
                render_cfg["degraded"] = apply_degradation
                render_cfg["mode"] = mode

                # 7. Render via Playwright
                html_doc = make_html_document(str(table_node), render_cfg)
                await page.set_content(html_doc, wait_until="load")
                measured = await measure_and_screenshot(page, out_png, table_rng, apply_degradation=apply_degradation)

                # 8. Assemble TSR Ground Truth
                cells_gt = []
                for i, (layout_item, ann, box) in enumerate(zip(cell_layout, cell_annotations, measured["cell_boxes_px"])):
                    text = ann["text"]
                    tokens = ann["tokens"]
                    cells_gt.append({
                        "cell_id": i,
                        "row_num": layout_item["row_start"],
                        "col_num": layout_item["col_start"],
                        "row_start": layout_item["row_start"],
                        "row_end": layout_item["row_end"],
                        "col_start": layout_item["col_start"],
                        "col_end": layout_item["col_end"],
                        "rowspan": layout_item["rowspan"],
                        "colspan": layout_item["colspan"],
                        "is_header": layout_item["is_header"],
                        "is_gibberish": ann["is_gibberish"],
                        "text": text,
                        "tokens": tokens,
                        "rendered_bbox_xyxy_px": box["xyxy"],
                        "rendered_bbox_xywh_px": box["xywh"]
                    })

                total_gibberish = sum(1 for c in cells_gt if c["is_gibberish"])
                gt_record = {
                    "image_id": image_id,
                    "image": f"images/{image_id}.png",
                    "source_id": source_id,
                    "mode": mode,
                    "complexity_bucket": sample.get("complexity_bucket", "unknown"),
                    "num_rows": metrics["rows"],
                    "num_cols": metrics["cols"],
                    "num_cells": len(cells_gt),
                    "num_gibberish_cells": total_gibberish,
                    "gibberish_replacement_ratio": round(total_gibberish / max(1, total_cells), 4),
                    "target_replacement_ratio": round(gibberish_prob, 4),
                    "max_word_len": max_word_len,
                    "render_config": render_cfg,
                    "html": {
                        "structure": {
                            "tokens": structure_tokens
                        },
                        "cells": cells_gt
                    }
                }

                ann_f.write(json.dumps(gt_record, ensure_ascii=False) + "\n")

                if draw_debug:
                    draw_debug_bboxes(out_png, cells_gt, out_debug_png)

                if (step + 1) % 25 == 0 or (step + 1) == len(indexed_samples):
                    logger.info(f"[{mode.upper()}] Rendered {step + 1}/{len(indexed_samples)} tables (Latest: {image_id}, {total_gibberish}/{total_cells} gibberish [{gibberish_prob*100:.1f}%])")

        await browser.close()


def cpu_render_worker(
    shard_id: int,
    indexed_samples: list,
    shared_configs: list,
    output_dir: Path,
    out_ann_path: Path,
    seed: int,
    mode: str,
    min_gibberish_prob: float,
    max_gibberish_prob: float,
    fixed_gibberish_prob: Optional[float],
    max_gibberish_cells_per_table: int,
    max_word_len: int,
    apply_degradation: bool,
    draw_debug: bool
):
    setup_logging(prefix=f"{mode.upper()}_Worker_{shard_id}")
    asyncio.run(async_render_shard(
        shard_id, indexed_samples, shared_configs, output_dir, out_ann_path,
        seed, mode, min_gibberish_prob, max_gibberish_prob, fixed_gibberish_prob,
        max_gibberish_cells_per_table, max_word_len, apply_degradation, draw_debug
    ))


# ==============================================================================
# DATASET GENERATION PIPELINE
# ==============================================================================

def run_dataset_generation(
    master_samples: list,
    shared_configs: list,
    output_dir: Path,
    mode: str,
    seed: int,
    workers: int,
    min_gibberish_prob: float,
    max_gibberish_prob: float,
    fixed_gibberish_prob: Optional[float],
    max_gibberish_cells_per_table: int,
    max_word_len: int,
    apply_degradation: bool,
    draw_debug: bool,
    split: str
):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = min(workers, len(master_samples))
    indexed_samples = list(enumerate(master_samples))
    shard_size = max(1, (len(indexed_samples) + num_workers - 1) // num_workers)
    shards = [indexed_samples[i:i + shard_size] for i in range(0, len(indexed_samples), shard_size)]

    desc = "100% of cells" if mode == "all" else (f"{fixed_gibberish_prob*100:.0f}% (Fixed)" if fixed_gibberish_prob is not None else f"{min_gibberish_prob*100:.0f}%-{max_gibberish_prob*100:.0f}% (Random)")
    LOG.info(f"[{mode.upper()}] Starting generation of {len(master_samples)} tables into {output_dir} (Gibberish replacement: {desc}, max word len: {max_word_len})...")

    ann_dir = output_dir / "annotations_shards"
    ann_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    part_paths = []
    for shard_id, shard_data in enumerate(shards):
        if not shard_data:
            continue
        out_shard_ann = ann_dir / f"shard_{shard_id:02d}.jsonl"
        part_paths.append(out_shard_ann)
        p = mp.Process(
            target=cpu_render_worker,
            args=(
                shard_id, shard_data, shared_configs, output_dir, out_shard_ann,
                seed, mode, min_gibberish_prob, max_gibberish_prob, fixed_gibberish_prob,
                max_gibberish_cells_per_table, max_word_len, apply_degradation, draw_debug
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge shard annotations
    final_ann_path = output_dir / "annotations.jsonl"
    all_records = []
    for part in part_paths:
        if part.exists():
            with part.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
            part.unlink(missing_ok=True)

    try:
        ann_dir.rmdir()
    except Exception:
        pass

    all_records.sort(key=lambda r: r["image_id"])
    with final_ann_path.open("w", encoding="utf-8") as out_f:
        for r in all_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info(f"[{mode.upper()}] Successfully generated {len(all_records)} tables! Annotations saved to: {final_ann_path}")

    # Build HTML preview card
    preview_cards = []
    for r in all_records[:50]:
        img_id = r["image_id"]
        source_id = r["source_id"]
        n_rows = r["num_rows"]
        n_cols = r["num_cols"]
        n_cells = r["num_cells"]
        n_gib = r["num_gibberish_cells"]
        ratio_pct = r.get("gibberish_replacement_ratio", 0) * 100
        style = r["render_config"]["style_family"]
        # Sample gibberish text from cells
        sample_gib_texts = [c["text"] for c in r["html"]["cells"] if c.get("is_gibberish") and c.get("text")][:4]
        sample_str = " | ".join(f'"{t}"' for t in sample_gib_texts)

        preview_cards.append(f"""
        <div style="margin-bottom: 30px; padding: 18px; border: 1px solid #ddd; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <h3 style="margin-top:0;">{img_id} <span style="font-weight:normal; font-size:14px; color:#666;">(Source: {source_id})</span></h3>
            <p><b>Dimensions:</b> {n_rows} rows &times; {n_cols} cols | <b>Cells:</b> {n_cells} | <b>Gibberish Injected:</b> {n_gib} ({ratio_pct:.1f}%) | <b>Style:</b> {style}</p>
            <p style="font-size: 13px; color: #444; background: #f8f9fa; padding: 6px 10px; border-radius: 4px;"><b>Sample Gibberish Cells:</b> {sample_str}</p>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <h4>Rendered Synthetic Table</h4>
                    <img src="images/{img_id}.png" style="max-width: 550px; border: 1px solid #ccc;" />
                </div>
                <div>
                    <h4>Bounding Boxes (Green=Original Text, Red=Gibberish, Blue=Header)</h4>
                    <img src="images_debug/{img_id}_debug.png" style="max-width: 550px; border: 1px solid #ccc;" />
                </div>
            </div>
        </div>
        """)

    preview_html_path = output_dir / "preview.html"
    title = f"Gibberish English Tables ({'All Cells' if mode == 'all' else 'Random Cells'}) Preview"
    preview_html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; background: #f4f6f8; }}
h1 {{ color: #1a1a1a; }}
</style>
</head>
<body>
<h1>{title} ({len(all_records)} Tables)</h1>
<p>Generated using PubTabNet evaluation split (<code>{split}</code>) with seed {seed}. Max word length strictly capped to {max_word_len} characters.</p>
{''.join(preview_cards)}
</body>
</html>"""
    with preview_html_path.open("w", encoding="utf-8") as pf:
        pf.write(preview_html_content)

    LOG.info(f"[{mode.upper()}] Visual preview card written to: {preview_html_path}")
    return final_ann_path


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    mp.set_start_method("spawn", force=True)
    setup_logging(prefix="MAIN")

    parser = argparse.ArgumentParser(description="Synthetic Gibberish English Table Generator from PubTabNet Seed Tables")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL, help="Path to PubTabNet jsonl")
    parser.add_argument("--split", type=str, default="val", help="PubTabNet split to sample from (default: val)")
    parser.add_argument("--mode", type=str, choices=["both", "random", "all"], default="both",
                        help="Mode: 'random' (random cells replaced), 'all' (all cells replaced), or 'both' (default: both)")
    parser.add_argument("--output-dir-random", type=Path, default=None, help="Output dir for random cells gibberish dataset")
    parser.add_argument("--output-dir-all", type=Path, default=None, help="Output dir for all cells gibberish dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir when running a single mode")
    parser.add_argument("--num-tables", type=int, default=500, help="Number of tables to generate (default: 500)")
    parser.add_argument("--min-gibberish-prob", type=float, default=0.40, help="Minimum gibberish replacement ratio for random mode (default: 0.40)")
    parser.add_argument("--max-gibberish-prob", type=float, default=0.60, help="Maximum gibberish replacement ratio for random mode (default: 0.60)")
    parser.add_argument("--fixed-gibberish-prob", type=float, default=None, help="Set exact fixed replacement ratio for random mode (e.g. 0.50)")
    parser.add_argument("--max-gibberish-cells-per-table", type=int, default=0, help="Cap on gibberish cells per table (0 = no cap)")
    parser.add_argument("--max-word-len", type=int, default=30, help="Maximum character length per gibberish word (default: 30)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel Playwright render workers (default: 8)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--degrade", action=argparse.BooleanOptionalAction, default=True, help="Apply printed document degradation")
    parser.add_argument("--no-debug-bboxes", action="store_true", help="Skip generating debug bounding box images")
    parser.add_argument("--sanity-check", action="store_true", help="Quick sanity check run with 5 tables")
    args = parser.parse_args()

    if args.sanity_check:
        args.num_tables = 5
        out_random = SCRIPT_DIR / "sanity_check_gibberish_random_cells"
        out_all = SCRIPT_DIR / "sanity_check_gibberish_all_cells"
    else:
        out_random = args.output_dir_random or (SCRIPT_DIR / "gibberish_random_tables_500")
        out_all = args.output_dir_all or (SCRIPT_DIR / "gibberish_all_tables_500")

    if args.output_dir and args.mode != "both":
        if args.mode == "random":
            out_random = args.output_dir
        else:
            out_all = args.output_dir

    cache_pool_file = DEFAULT_CACHE_DIR / f"seed_master_pool_{args.split}_{args.num_tables}_seed{args.seed}.json"

    # 1. Load exact master seed tables from eval split
    LOG.info(f"Generating master pool of {args.num_tables} tables from split '{args.split}' with seed {args.seed}...")
    master_samples = generate_global_pools(
        input_jsonl=args.input_jsonl,
        num_tables=args.num_tables,
        seed=args.seed,
        split=args.split,
        cache_file=cache_pool_file,
        is_sanity_check=args.sanity_check
    )

    # 2. Generate deterministic style configurations (35% MUSTARD stressors)
    rng = random.Random(args.seed)
    stressor_styles = ["borderless_dense", "underlined_header", "inverse_banner"]
    shared_configs = []
    for idx in range(len(master_samples)):
        if idx < max(3, int(len(master_samples) * 0.35)):
            chosen_style = stressor_styles[idx % len(stressor_styles)]
        else:
            chosen_style = rng.choice(STYLE_NAMES)

        shared_configs.append({
            "style_family": chosen_style,
            "background_variant": rng.choice(COLOR_PALETTE_NAMES) if (chosen_style != "inverse_banner" and rng.random() < COLOR_FILL_PROBABILITY) else "none",
            "bold_header": rng.random() < 0.40,
            "font_family": rng.choice(FONTS),
            "font_size_px": rng.choice([10, 11, 12, 13]),
            "font_weight": rng.choice([400, 500]),
            "line_height": round(rng.uniform(1.15, 1.4), 2),
            "cell_padding_px": rng.randint(2, 5) if chosen_style == "borderless_dense" else rng.choice([4, 6, 8])
        })

    # 3. Execute Generation
    modes_to_run = ["random", "all"] if args.mode == "both" else [args.mode]

    for current_mode in modes_to_run:
        cur_out_dir = out_random if current_mode == "random" else out_all
        run_dataset_generation(
            master_samples=master_samples,
            shared_configs=shared_configs,
            output_dir=cur_out_dir,
            mode=current_mode,
            seed=args.seed,
            workers=args.workers,
            min_gibberish_prob=args.min_gibberish_prob,
            max_gibberish_prob=args.max_gibberish_prob,
            fixed_gibberish_prob=args.fixed_gibberish_prob,
            max_gibberish_cells_per_table=args.max_gibberish_cells_per_table,
            max_word_len=args.max_word_len,
            apply_degradation=args.degrade,
            draw_debug=not args.no_debug_bboxes,
            split=args.split
        )

    LOG.info("All requested dataset generations completed successfully!")


if __name__ == "__main__":
    main()
