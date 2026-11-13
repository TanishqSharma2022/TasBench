#!/usr/bin/env python3
"""Synthetic Chemical Table Generator Combining Both OCSR_BENCHMARK Molecule Images and ORD SMILES.

Key Features:
- Exact same seed (default: 2026) and same master seed tables as generate_chemical_tables.py.
- Stratified sampling: 30% Simple, 30% Medium, 40% Complex.
- MUSTARD-style layout stressors (borderless_dense, underlined_header, inverse_banner) on 35% of tables.
- Printed document degradation (ink bleed, streaks, toner speckles, smudging, soft blur, JPEG compression).
- Combines multimodal chemical representations:
    * Molecule diagram images sampled from OCSR_BENCHMARK (ACS, ChemDraw, USPTO, CLEF, etc.)
    * Chemical SMILES text strings sampled from ORD (Open Reaction Database) projections.
- Supports configurable image-to-SMILES ratio (--img-ratio, default: 0.50) and combination modes:
    * "mixed": randomly intersperses molecule diagram cells and SMILES text cells within the table.
    * "both_in_cell": injects both the molecule diagram image AND the SMILES formula into the same cell.
- Color-coded debug bounding boxes: Crimson (Image), Dark Orange (SMILES), Purple (Both), Blue (Header), Green (Text).
- Outputs TSR bounding boxes, molecule metadata, SMILES, structure tokens, and preview HTML.
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
import base64
import copy
import html as html_lib
import io
import json
import logging
import multiprocessing as mp
import random
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from bs4 import BeautifulSoup, NavigableString
from PIL import Image as PILImage, ImageDraw
import pyarrow.parquet as pq

LOG = logging.getLogger("chemical_combined_tables")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSONL = Path.home() / "scratch/tanishq/table-understanding/data/Table Datasets/pubtabnet/PubTabNet_2.0.0.jsonl"
DEFAULT_OCSR_DIR = SCRIPT_DIR / "OCSR_BENCHMARK"
DEFAULT_ORD_PATH = SCRIPT_DIR / "ord_data/projections_10000.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "chemical_tables_combined_500"
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


def setup_logging(prefix: str = "CHEM_COMBINED_TSR") -> None:
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
# MOLECULE POOL (OCSR_BENCHMARK IMAGES)
# ==============================================================================

class MoleculePool:
    """Manages fast in-memory cached sampling of molecule images and SMILES from OCSR_BENCHMARK."""

    def __init__(self, ocsr_dir: Path):
        self.ocsr_dir = Path(ocsr_dir)
        if not self.ocsr_dir.exists():
            raise FileNotFoundError(f"OCSR benchmark directory not found: {self.ocsr_dir}")

        self.datasets: Dict[str, List[Path]] = {}
        self.dataset_weights: Dict[str, int] = {}
        self.parquet_files: Dict[Path, pq.ParquetFile] = {}
        self.row_group_cache: Dict[Tuple[str, int], Any] = {}
        self._init_benchmark()

    def _init_benchmark(self):
        for sub in sorted(self.ocsr_dir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                parquets = list(sub.glob("*.parquet"))
                if parquets:
                    total_rows = 0
                    for p in parquets:
                        try:
                            pf = pq.ParquetFile(p)
                            self.parquet_files[p] = pf
                            total_rows += pf.metadata.num_rows
                        except Exception as e:
                            LOG.warning(f"Could not open parquet {p}: {e}")
                    if total_rows > 0:
                        self.datasets[sub.name] = parquets
                        self.dataset_weights[sub.name] = total_rows

        LOG.info(f"Loaded {len(self.datasets)} OCSR datasets: {list(self.datasets.keys())} (Total molecules: {sum(self.dataset_weights.values()):,})")

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        """Samples a molecule record with image bytes, base64 data URI, SMILES, and image_id."""
        ds_names = list(self.datasets.keys())
        ds_name = rng.choice(ds_names)
        parquet_path = rng.choice(self.datasets[ds_name])
        pf = self.parquet_files[parquet_path]

        rg_idx = rng.randint(0, pf.num_row_groups - 1)
        cache_key = (str(parquet_path), rg_idx)
        if cache_key not in self.row_group_cache:
            rg_table = pf.read_row_group(rg_idx, columns=["image_id", "image", "SMILES"])
            self.row_group_cache[cache_key] = rg_table
        else:
            rg_table = self.row_group_cache[cache_key]

        local_idx = rng.randint(0, rg_table.num_rows - 1)
        row = rg_table.slice(local_idx, 1).to_pydict()

        image_id = str(row["image_id"][0])
        smiles = str(row["SMILES"][0])
        img_dict = row["image"][0]
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) and "bytes" in img_dict else b""
        b64_str = base64.b64encode(img_bytes).decode("ascii")

        return {
            "source": "OCSR_BENCHMARK",
            "dataset": ds_name,
            "image_id": image_id,
            "smiles": smiles,
            "image_bytes": img_bytes,
            "data_uri": f"data:image/png;base64,{b64_str}"
        }


# ==============================================================================
# ORD SMILES POOL
# ==============================================================================

class OrdSmilesPool:
    """Manages fast in-memory sampling of SMILES strings from ORD projections dataset.
    
    Strictly caps SMILES length (e.g. 40-50 characters) to prevent table columns
    from stretching excessively.
    """

    def __init__(self, ord_path: Path, max_smiles_len: int = 45, min_smiles_len: int = 3):
        self.ord_path = Path(ord_path)
        self.max_smiles_len = max_smiles_len
        self.min_smiles_len = min_smiles_len
        if not self.ord_path.exists():
            LOG.warning(f"ORD projections file not found at {self.ord_path}. Attempting download...")
            from download_ord_data import download_ord_projections
            download_ord_projections(self.ord_path, limit=10_000)

        self.smiles_items: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        count = 0
        seen_smiles = set()
        with self.ord_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rxn_id = row.get("reaction_id", f"ord_{count:06d}")
                    rxn_smiles = row.get("smiles", "")
                    if not rxn_smiles:
                        continue

                    components = [
                        c.strip() for c in rxn_smiles.replace(">>", ".").replace(">", ".").split(".")
                        if self.min_smiles_len <= len(c.strip()) <= self.max_smiles_len
                    ]

                    for c in components:
                        if c not in seen_smiles:
                            seen_smiles.add(c)
                            self.smiles_items.append({
                                "source": "ORD",
                                "reaction_id": rxn_id,
                                "smiles": c,
                                "smiles_type": "molecule",
                            })

                    if self.min_smiles_len <= len(rxn_smiles) <= self.max_smiles_len and rxn_smiles not in seen_smiles:
                        seen_smiles.add(rxn_smiles)
                        self.smiles_items.append({
                            "source": "ORD",
                            "reaction_id": rxn_id,
                            "smiles": rxn_smiles,
                            "smiles_type": "reaction",
                        })

                    count += 1
                except Exception:
                    continue

        if not self.smiles_items:
            raise ValueError(f"No valid SMILES found with length <= {self.max_smiles_len} in ORD dataset at: {self.ord_path}")

        LOG.info(f"Loaded {len(self.smiles_items):,} unique ORD SMILES (strictly capped at {self.min_smiles_len}-{self.max_smiles_len} chars)")

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        """Samples a SMILES entry strictly satisfying the character cap."""
        return self.smiles_items[rng.randint(0, len(self.smiles_items) - 1)]


# ==============================================================================
# SEED TABLE EXTRACTION
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
    LOG.info(f"Scanning {input_jsonl} (split: {valid_splits}) for {caps['simple']} Simple (30%) / {caps['medium']} Medium (30%) / {caps['complex']} Complex (40%)...")

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
td, th {{ vertical-align: middle; overflow-wrap: anywhere; word-break: normal; padding: {render_cfg['cell_padding_px']}px; }}
img.chem-mol {{
    display: block;
    margin: 2px auto;
    max-height: 75px;
    max-width: 140px;
    width: auto;
    height: auto;
    object-fit: contain;
    background: transparent;
}}
.chem-smiles {{
    display: inline-block;
    word-break: break-all;
    overflow-wrap: anywhere;
    font-family: inherit;
    font-size: 0.92em;
    line-height: 1.25;
    letter-spacing: 0.01em;
}}
.chem-combo-cell {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
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
    """Draws color-coded bounding boxes on table image:

    - Crimson: Molecule Diagram Image
    - Dark Orange: SMILES Text String
    - Purple: Both Image & SMILES in same cell
    - Dodger Blue: Table Header
    - Forest Green: Regular PubTabNet Text
    """
    with PILImage.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        for cell in cells_gt:
            x0, y0, x1, y1 = cell["rendered_bbox_xyxy_px"]
            chem_type = cell.get("chem_type")
            if chem_type == "both":
                draw.rectangle([x0, y0, x1, y1], outline=(148, 0, 211), width=3)
            elif chem_type == "image":
                draw.rectangle([x0, y0, x1, y1], outline=(220, 20, 60), width=3)
            elif chem_type == "smiles":
                draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0), width=3)
            elif cell["is_header"]:
                draw.rectangle([x0, y0, x1, y1], outline=(30, 144, 255), width=2)
            else:
                draw.rectangle([x0, y0, x1, y1], outline=(34, 139, 34), width=1)
        img.save(out_debug_path, format="PNG")


# ==============================================================================
# ASYNC RENDERING WORKER (Sharded for fast parallel execution)
# ==============================================================================

async def async_render_shard(
    shard_id: int,
    indexed_samples: List[Tuple[int, dict]],
    shared_configs: List[dict],
    ocsr_dir: Path,
    ord_path: Path,
    output_dir: Path,
    out_ann_path: Path,
    seed: int,
    min_mol_prob: float,
    max_mol_prob: float,
    fixed_mol_prob: Optional[float],
    max_molecules_per_table: int,
    max_smiles_len: int,
    img_ratio: float,
    combine_mode: str,
    apply_degradation: bool,
    draw_debug: bool
):
    from playwright.async_api import async_playwright

    logger = logging.getLogger(f"COMBINED_SHARD_{shard_id}")
    images_dir = output_dir / "images"
    debug_dir = output_dir / "images_debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    if draw_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)

    mol_pool = MoleculePool(ocsr_dir)
    smiles_pool = OrdSmilesPool(ord_path, max_smiles_len=max_smiles_len)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 2400, "height": 2400}, device_scale_factor=3)

        with out_ann_path.open("w", encoding="utf-8") as ann_f:
            for step, (idx, sample) in enumerate(indexed_samples):
                record = sample["record"]
                source_id = sample["source_id"]
                image_id = f"chem_combined_table_{idx:05d}"
                out_png = images_dir / f"{image_id}.png"
                out_debug_png = debug_dir / f"{image_id}_debug.png"

                table_rng = random.Random(seed + idx)

                # 1. Build initial soup from PubTabNet text
                src_texts = [source_cell_text(c) for c in source_cells(record)]
                soup, table_node, _, _ = build_translated_soup(record, src_texts)

                # 2. Extract cell layout & determine headers vs body
                metrics, cell_layout = extract_cell_layout(table_node)
                total_cells = metrics["cells"]

                # 3. Determine Chemical Cell Replacement Ratio (Fixed 40% vs Random 30%-60%)
                if fixed_mol_prob is not None:
                    mol_prob = fixed_mol_prob
                else:
                    mol_prob = table_rng.uniform(min_mol_prob, max_mol_prob)

                num_to_inject = max(1, int(round(total_cells * mol_prob)))
                if max_molecules_per_table and max_molecules_per_table > 0:
                    num_to_inject = min(num_to_inject, max_molecules_per_table)

                body_indices = [i for i, c in enumerate(cell_layout) if not c["is_header"]]
                if len(body_indices) >= num_to_inject:
                    chosen_indices = set(table_rng.sample(body_indices, num_to_inject))
                else:
                    remaining = num_to_inject - len(body_indices)
                    other_indices = [i for i in range(total_cells) if i not in set(body_indices)]
                    chosen_indices = set(body_indices) | set(table_rng.sample(other_indices, min(remaining, len(other_indices))))

                # 4. Inject combined chemicals (Images from OCSR and/or SMILES from ORD)
                cell_annotations = []
                num_images = 0
                num_smiles = 0

                for i, cell_info in enumerate(cell_layout):
                    dom_cell = cell_info["cell_node"]
                    if i in chosen_indices:
                        if combine_mode == "both_in_cell":
                            # Combined: Both image and SMILES in same cell
                            mol = mol_pool.sample(table_rng)
                            smiles_item = smiles_pool.sample(table_rng)
                            dom_cell.clear()

                            combo_div = soup.new_tag("div")
                            combo_div["class"] = "chem-combo-cell"

                            img_tag = soup.new_tag("img", src=mol["data_uri"], alt=mol["smiles"])
                            img_tag["class"] = "chem-mol"
                            combo_div.append(img_tag)

                            span_tag = soup.new_tag("span")
                            span_tag["class"] = "chem-smiles"
                            span_tag.string = smiles_item["smiles"]
                            combo_div.append(span_tag)

                            dom_cell.append(combo_div)
                            num_images += 1
                            num_smiles += 1

                            cell_annotations.append({
                                "is_molecule": True,
                                "is_molecule_image": True,
                                "is_smiles": True,
                                "chem_type": "both",
                                "smiles": smiles_item["smiles"],
                                "text": f"{mol['smiles']} {smiles_item['smiles']}",
                                "molecule_metadata": {
                                    "image_metadata": {
                                        "source": "OCSR_BENCHMARK",
                                        "dataset": mol["dataset"],
                                        "image_id": mol["image_id"],
                                        "smiles": mol["smiles"]
                                    },
                                    "smiles_metadata": {
                                        "source": "ORD",
                                        "reaction_id": smiles_item["reaction_id"],
                                        "smiles": smiles_item["smiles"],
                                        "smiles_type": smiles_item["smiles_type"]
                                    }
                                }
                            })
                        else:
                            # Mixed mode: allocate between image diagram and SMILES text
                            use_image = (table_rng.random() < img_ratio)
                            dom_cell.clear()

                            if use_image:
                                mol = mol_pool.sample(table_rng)
                                img_tag = soup.new_tag("img", src=mol["data_uri"], alt=mol["smiles"])
                                img_tag["class"] = "chem-mol"
                                dom_cell.append(img_tag)
                                num_images += 1

                                cell_annotations.append({
                                    "is_molecule": True,
                                    "is_molecule_image": True,
                                    "is_smiles": False,
                                    "chem_type": "image",
                                    "smiles": mol["smiles"],
                                    "text": mol["smiles"],
                                    "molecule_metadata": {
                                        "source": "OCSR_BENCHMARK",
                                        "dataset": mol["dataset"],
                                        "image_id": mol["image_id"],
                                        "smiles": mol["smiles"]
                                    }
                                })
                            else:
                                smiles_item = smiles_pool.sample(table_rng)
                                span_tag = soup.new_tag("span")
                                span_tag["class"] = "chem-smiles"
                                span_tag.string = smiles_item["smiles"]
                                dom_cell.append(span_tag)
                                num_smiles += 1

                                cell_annotations.append({
                                    "is_molecule": True,
                                    "is_molecule_image": False,
                                    "is_smiles": True,
                                    "chem_type": "smiles",
                                    "smiles": smiles_item["smiles"],
                                    "text": smiles_item["smiles"],
                                    "molecule_metadata": {
                                        "source": "ORD",
                                        "reaction_id": smiles_item["reaction_id"],
                                        "smiles": smiles_item["smiles"],
                                        "smiles_type": smiles_item["smiles_type"]
                                    }
                                })
                    else:
                        cell_annotations.append({
                            "is_molecule": False,
                            "is_molecule_image": False,
                            "is_smiles": False,
                            "chem_type": "text",
                            "smiles": None,
                            "text": cell_info["original_text"],
                            "molecule_metadata": None
                        })

                # 5. Extract structure tokens for TSR
                structure_tokens = extract_structure_tokens_from_table(table_node)

                # 6. Render style configuration
                render_cfg = shared_configs[idx]
                render_cfg["degraded"] = apply_degradation

                # 7. Render via Playwright
                html_doc = make_html_document(str(table_node), render_cfg)
                await page.set_content(html_doc, wait_until="load")
                measured = await measure_and_screenshot(page, out_png, table_rng, apply_degradation=apply_degradation)

                # 8. Assemble TSR Ground Truth
                cells_gt = []
                for i, (layout_item, ann, box) in enumerate(zip(cell_layout, cell_annotations, measured["cell_boxes_px"])):
                    text = ann["text"]
                    tokens = [text] if text else []
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
                        "is_molecule": ann["is_molecule"],
                        "is_molecule_image": ann["is_molecule_image"],
                        "is_smiles": ann["is_smiles"],
                        "chem_type": ann["chem_type"],
                        "text": text,
                        "tokens": tokens,
                        "molecule_metadata": ann["molecule_metadata"],
                        "rendered_bbox_xyxy_px": box["xyxy"],
                        "rendered_bbox_xywh_px": box["xywh"]
                    })

                gt_record = {
                    "image_id": image_id,
                    "image": f"images/{image_id}.png",
                    "source_id": source_id,
                    "complexity_bucket": sample.get("complexity_bucket", "unknown"),
                    "num_rows": metrics["rows"],
                    "num_cols": metrics["cols"],
                    "num_cells": len(cells_gt),
                    "num_molecules_total": len(chosen_indices),
                    "num_molecule_images": num_images,
                    "num_smiles": num_smiles,
                    "combine_mode": combine_mode,
                    "molecule_replacement_ratio": round(len(chosen_indices) / max(1, total_cells), 4),
                    "target_replacement_ratio": round(mol_prob, 4),
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
                    logger.info(f"Rendered {step + 1}/{len(indexed_samples)} tables (Latest: {image_id}, {num_images} images + {num_smiles} SMILES [{mol_prob*100:.1f}%])")

        await browser.close()


def cpu_render_worker(
    shard_id: int,
    indexed_samples: list,
    shared_configs: list,
    ocsr_dir: Path,
    ord_path: Path,
    output_dir: Path,
    out_ann_path: Path,
    seed: int,
    min_mol_prob: float,
    max_mol_prob: float,
    fixed_mol_prob: Optional[float],
    max_molecules_per_table: int,
    max_smiles_len: int,
    img_ratio: float,
    combine_mode: str,
    apply_degradation: bool,
    draw_debug: bool
):
    setup_logging(prefix=f"Worker {shard_id}")
    asyncio.run(async_render_shard(
        shard_id, indexed_samples, shared_configs, ocsr_dir, ord_path, output_dir, out_ann_path,
        seed, min_mol_prob, max_mol_prob, fixed_mol_prob, max_molecules_per_table,
        max_smiles_len, img_ratio, combine_mode, apply_degradation, draw_debug
    ))


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    mp.set_start_method("spawn", force=True)
    setup_logging(prefix="MAIN")

    parser = argparse.ArgumentParser(description="Synthetic Chemical Table TSR Generator Combining OCSR Benchmark Images and ORD SMILES")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL, help="Path to PubTabNet jsonl")
    parser.add_argument("--split", type=str, default="val", help="PubTabNet split to sample from (default: val)")
    parser.add_argument("--ocsr-dir", type=Path, default=DEFAULT_OCSR_DIR, help="Path to OCSR_BENCHMARK folder")
    parser.add_argument("--ord-path", type=Path, default=DEFAULT_ORD_PATH, help="Path to ORD projections jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--num-tables", type=int, default=500, help="Number of chemical tables to generate (default: 500)")
    parser.add_argument("--min-mol-prob", type=float, default=0.30, help="Minimum chemical replacement ratio (default: 0.30)")
    parser.add_argument("--max-mol-prob", type=float, default=0.60, help="Maximum chemical replacement ratio (default: 0.60)")
    parser.add_argument("--fixed-mol-prob", type=float, default=None, help="Set exact fixed replacement ratio (e.g. 0.40 for 40 percent replacement)")
    parser.add_argument("--max-molecules-per-table", type=int, default=0, help="Optional cap on chemicals per table (0 = no cap)")
    parser.add_argument("--max-smiles-len", type=int, default=45, help="Maximum character length of SMILES string (default: 45, max recommended: 50)")
    parser.add_argument("--img-ratio", type=float, default=0.50, help="Proportion of injected chemical cells that are diagram images vs SMILES (default: 0.50)")
    parser.add_argument("--combine-mode", type=str, default="mixed", choices=["mixed", "both_in_cell"], help="Combine mode: 'mixed' (cells alternate image/SMILES) or 'both_in_cell' (each cell has both)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel Playwright render workers (default: 8)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--degrade", action=argparse.BooleanOptionalAction, default=True, help="Apply printed document degradation")
    parser.add_argument("--no-debug-bboxes", action="store_true", help="Skip generating debug bounding box images")
    parser.add_argument("--sanity-check", action="store_true", help="Quick run with 5 tables")
    args = parser.parse_args()

    if args.sanity_check:
        args.num_tables = 5
        args.output_dir = DEFAULT_CACHE_DIR / "sanity_check_chemical_tables_combined"

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    # 3. Shard work across parallel CPU workers
    num_workers = min(args.workers, len(master_samples))
    indexed_samples = list(enumerate(master_samples))
    shard_size = max(1, (len(indexed_samples) + num_workers - 1) // num_workers)
    shards = [indexed_samples[i:i + shard_size] for i in range(0, len(indexed_samples), shard_size)]

    prob_desc = f"{args.fixed_mol_prob*100:.0f}% (Fixed)" if args.fixed_mol_prob is not None else f"{args.min_mol_prob*100:.0f}%-{args.max_mol_prob*100:.0f}% (Random)"
    LOG.info(f"Launching {len(shards)} workers to render {len(master_samples)} chemical combined tables (Replacement: {prob_desc}, Mode: {args.combine_mode}, Image ratio: {args.img_ratio*100:.0f}%)...")

    ann_dir = args.output_dir / "annotations_shards"
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
                shard_id, shard_data, shared_configs, args.ocsr_dir, args.ord_path, args.output_dir, out_shard_ann,
                args.seed, args.min_mol_prob, args.max_mol_prob, args.fixed_mol_prob, args.max_molecules_per_table,
                args.max_smiles_len, args.img_ratio, args.combine_mode, args.degrade, not args.no_debug_bboxes
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 4. Merge shard annotations in exact index order
    final_ann_path = args.output_dir / "annotations.jsonl"
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

    # Sort deterministically by image_id
    all_records.sort(key=lambda r: r["image_id"])
    with final_ann_path.open("w", encoding="utf-8") as out_f:
        for r in all_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOG.info(f"Successfully generated {len(all_records)} combined chemical tables! Annotations saved to: {final_ann_path}")

    # 5. Build HTML preview card
    preview_cards = []
    for r in all_records[:50]:
        img_id = r["image_id"]
        source_id = r["source_id"]
        n_rows = r["num_rows"]
        n_cols = r["num_cols"]
        n_cells = r["num_cells"]
        n_mols = r.get("num_molecules_total", 0)
        n_imgs = r.get("num_molecule_images", 0)
        n_smiles = r.get("num_smiles", 0)
        ratio_pct = r.get("molecule_replacement_ratio", 0) * 100
        style = r["render_config"]["style_family"]
        preview_cards.append(f"""
        <div style="margin-bottom: 30px; padding: 18px; border: 1px solid #ddd; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <h3 style="margin-top:0;">{img_id} <span style="font-weight:normal; font-size:14px; color:#666;">(PubTabNet Source: {source_id})</span></h3>
            <p><b>Dimensions:</b> {n_rows} rows &times; {n_cols} cols | <b>Cells:</b> {n_cells} | <b>Chemicals:</b> {n_mols} ({ratio_pct:.1f}%) [{n_imgs} Images, {n_smiles} SMILES] | <b>Style:</b> {style}</p>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <h4>Synthetic Combined Chemical Table</h4>
                    <img src="images/{img_id}.png" style="max-width: 550px; border: 1px solid #ccc;" />
                </div>
                <div>
                    <h4>TSR Bounding Boxes (Crimson=Image, Dark Orange=SMILES, Purple=Both, Blue=Header, Green=Text)</h4>
                    <img src="images_debug/{img_id}_debug.png" style="max-width: 550px; border: 1px solid #ccc;" />
                </div>
            </div>
        </div>
        """)

    preview_html_path = args.output_dir / "preview.html"
    preview_html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Chemical Combined Tables TSR Dataset Preview</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 24px; background: #f4f6f8; }}
h1 {{ color: #1a1a1a; }}
</style>
</head>
<body>
<h1>Chemical Combined Tables TSR Dataset Preview ({len(all_records)} Tables)</h1>
<p>Generated combining OCSR_BENCHMARK molecule diagram images and ORD SMILES text, matching seed {args.seed}.</p>
{''.join(preview_cards)}
</body>
</html>"""
    with preview_html_path.open("w", encoding="utf-8") as pf:
        pf.write(preview_html_content)

    LOG.info(f"Visual preview card written to: {preview_html_path}")


if __name__ == "__main__":
    main()
