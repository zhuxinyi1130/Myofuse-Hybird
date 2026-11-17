#!/usr/bin/env python3
"""Desktop application for manual review of MyoFuse-Hybrid results.

An image canvas occupies the main area, analysis information is shown on the
right, the mouse wheel controls zoom, the middle button pans, left click
changes a nucleus, and right click excludes a nucleus. The application operates
on the outputs produced by this project.

Automatic predictions are never overwritten.  Manual decisions are exported
as separate columns and as an audit log so that automatic and corrected fusion
indices remain distinguishable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageTk
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from tifffile import imread, imwrite


IN = 1
OUT = 0
EXCLUDE = -1
STATE_NAMES = {IN: "In", OUT: "Out", EXCLUDE: "Exclude"}
STATE_COLORS = {
    IN: np.array([38, 222, 95], dtype=np.uint8),
    OUT: np.array([255, 77, 79], dtype=np.uint8),
    EXCLUDE: np.array([180, 180, 180], dtype=np.uint8),
}
REVIEW_CANDIDATE_COLOR = np.array([255, 220, 0], dtype=np.uint8)
# A manually added nucleus does not exist before it is added and after it is
# removed. ABSENT marks those two ends of a reversible add/remove change.
ABSENT = -99


def state_name(state: int) -> str:
    return "Absent" if state == ABSENT else STATE_NAMES[state]


def utc_now() -> str:
    """Return an ISO timestamp suitable for an audit record."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def scale_channel(image: np.ndarray) -> np.ndarray:
    """Convert one microscopy channel to display-ready 8-bit intensities."""
    data = image.astype(np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99.7])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((data - low) / (high - low), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def choose_class_column(frame: pd.DataFrame) -> str:
    """Select the automatic class column from old and current result files."""
    for name in (
        "Automatic final class",
        "Automatic Final Class",
        "Final Class",
        "Original Class",
    ):
        if name in frame.columns:
            return name
    raise ValueError(
        "The Per nucleus sheet has no supported automatic class column. "
        "Expected Automatic final class, Final Class, or Original Class."
    )


def infer_stem(results_path: Path) -> str:
    """Recover the image/scene stem used by the analysis output files."""
    suffixes = (
        "__hybrid_nucleus_aware_results",
        "_hybrid_nucleus_aware_results",
    )
    stem = results_path.stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_related_file(results_path: Path, names: list[str]) -> Optional[Path]:
    """Search nearby output folders for a uniquely named related file."""
    roots = [results_path.parent, results_path.parent.parent]
    for root in roots:
        if not root.exists():
            continue
        candidates: list[Path] = []
        for name in names:
            candidates.extend(root.rglob(name))
        unique = sorted({path.resolve() for path in candidates if path.is_file()})
        # Prefer an exact companion in the result workbook's own folder. A
        # broader search may contain valid files from several analysis runs.
        direct = [path for path in unique if path.parent == results_path.parent.resolve()]
        if len(direct) == 1:
            return direct[0]
        if len(unique) == 1:
            return unique[0]
    return None


@dataclass
class Change:
    """One reversible manual decision."""

    label_id: int
    old_state: int
    new_state: int
    action: str
    timestamp: str = field(default_factory=utc_now)


@dataclass
class ReviewProject:
    """Data and export logic for one reviewed image scene."""

    results_path: Path
    label_mask_path: Path
    czi_path: Optional[Path] = None
    scene: Optional[str] = None
    nuclei_channel: int = 0
    actin_channel: int = 1
    mf20_channel: int = 2
    channel_cache: Optional[Path] = None
    myotube_mask_path: Optional[Path] = None

    sheets: dict[str, pd.DataFrame] = field(init=False, default_factory=dict)
    per_nucleus: pd.DataFrame = field(init=False)
    labels: np.ndarray = field(init=False)
    automatic: dict[int, int] = field(init=False, default_factory=dict)
    current: dict[int, int] = field(init=False, default_factory=dict)
    centres: dict[int, tuple[float, float]] = field(init=False, default_factory=dict)
    history: list[Change] = field(init=False, default_factory=list)
    redo_stack: list[Change] = field(init=False, default_factory=list)
    channels: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    fallback_background: Optional[np.ndarray] = field(init=False, default=None)
    myotube_mask: Optional[np.ndarray] = field(init=False, default=None)
    class_column: str = field(init=False)
    review_candidates: set[int] = field(init=False, default_factory=set)
    # Nuclei placed by the reviewer where segmentation found none:
    # label id -> (x, y, radius). Geometry is kept after removal so undo/redo
    # can restore it; a nucleus is active only while it is in self.current.
    added: dict[int, tuple[float, float, float]] = field(init=False, default_factory=dict)
    typical_radius: float = field(init=False, default=6.0)

    def __post_init__(self) -> None:
        self.results_path = self.results_path.expanduser().resolve()
        self.label_mask_path = self.label_mask_path.expanduser().resolve()
        if self.czi_path is not None:
            self.czi_path = self.czi_path.expanduser().resolve()
        if self.channel_cache is not None:
            self.channel_cache = self.channel_cache.expanduser().resolve()
        if self.myotube_mask_path is not None:
            self.myotube_mask_path = self.myotube_mask_path.expanduser().resolve()
        self._load_tables()
        self._load_masks()
        self._load_background()

    def _load_tables(self) -> None:
        book = pd.ExcelFile(self.results_path)
        self.sheets = {name: pd.read_excel(self.results_path, sheet_name=name) for name in book.sheet_names}
        if "Per nucleus" not in self.sheets:
            raise ValueError("Result workbook must contain a 'Per nucleus' sheet")
        self.per_nucleus = self.sheets["Per nucleus"].copy()
        if "Label ID" not in self.per_nucleus.columns:
            raise ValueError("The Per nucleus sheet must contain Label ID")
        if self.per_nucleus["Label ID"].duplicated().any():
            raise ValueError("Label ID values must be unique")

        self.class_column = choose_class_column(self.per_nucleus)
        ids = self.per_nucleus["Label ID"].astype(int)
        classes = self.per_nucleus[self.class_column].astype(int)
        if not classes.isin([OUT, IN]).all():
            raise ValueError(f"{self.class_column} must contain only 0 or 1")
        self.automatic = dict(zip(ids, classes))
        self.current = dict(self.automatic)
        candidate_columns = [
            name for name in (
                "Direction-aware review candidate",
                "Graph-aware review candidate",
                "Weak-signal review candidate",
                "Adaptive weak-signal review candidate",
                "Actin structure Hybrid-anchor review candidate",
                "Low-MyoFuse support review candidate",
                "High-probability neighbor review candidate",
                "Learned model review candidate",
            )
            if name in self.per_nucleus.columns
        ]
        if candidate_columns:
            candidate_mask = self.per_nucleus[candidate_columns].fillna(0).astype(bool).any(axis=1)
            self.review_candidates = set(ids[candidate_mask].astype(int))

        if {"X", "Y"}.issubset(self.per_nucleus.columns):
            self.centres = {
                int(row["Label ID"]): (float(row["X"]), float(row["Y"]))
                for _, row in self.per_nucleus.iterrows()
            }

        if self.scene is None and "Summary" in self.sheets and "Scene" in self.sheets["Summary"].columns:
            value = self.sheets["Summary"].iloc[0]["Scene"]
            if pd.notna(value):
                self.scene = str(value)
        self.scene = self.scene or "Image:0"
        self._read_channel_parameters()

    def _read_channel_parameters(self) -> None:
        parameters = self.sheets.get("Parameters")
        if parameters is None or not {"Parameter", "Value"}.issubset(parameters.columns):
            return
        values = dict(zip(parameters["Parameter"].astype(str), parameters["Value"].astype(str)))
        mapping = {
            "Nuclei channel": "nuclei_channel",
            "Actin channel": "actin_channel",
            "MF20 channel": "mf20_channel",
        }
        for key, attribute in mapping.items():
            text = values.get(key, "")
            if text.startswith("C") and ":" in text:
                try:
                    setattr(self, attribute, int(text[1 : text.index(":")]))
                except ValueError:
                    pass

    def _load_masks(self) -> None:
        self.labels = imread(self.label_mask_path).astype(np.int32)
        if self.labels.ndim != 2:
            raise ValueError("The label mask must be a two-dimensional TIFF")
        label_ids = set(np.unique(self.labels)) - {0}
        table_ids = set(self.automatic)
        if label_ids != table_ids:
            missing_table = sorted(label_ids - table_ids)[:10]
            missing_mask = sorted(table_ids - label_ids)[:10]
            raise ValueError(
                "Label mask and workbook IDs differ. "
                f"Only in mask: {missing_table}; only in workbook: {missing_mask}"
            )
        areas = np.bincount(self.labels.ravel())[1:]
        areas = areas[areas > 0]
        if areas.size:
            self.typical_radius = float(np.sqrt(np.median(areas) / np.pi))
        if not self.centres:
            for label_id in sorted(label_ids):
                y, x = np.argwhere(self.labels == label_id).mean(axis=0)
                self.centres[int(label_id)] = (float(x), float(y))

        stem = infer_stem(self.results_path)
        mask_path = self.myotube_mask_path or find_related_file(
            self.results_path,
            [
                f"{stem}__hybrid_nucleus_aware_mask.tif",
                f"{stem}_hybrid_nucleus_aware_mask.tif",
            ],
        )
        if mask_path is not None:
            mask = imread(mask_path)
            if mask.shape == self.labels.shape:
                self.myotube_mask = mask > 0
                self.myotube_mask_path = mask_path

    def _load_background(self) -> None:
        if self.channel_cache is not None and self.channel_cache.is_file():
            self._load_channel_cache(self.channel_cache)
            return
        if self.czi_path is not None:
            try:
                cache = self._ensure_channel_cache()
                self._load_channel_cache(cache)
            except (ModuleNotFoundError, FileNotFoundError, subprocess.CalledProcessError) as error:
                print(f"CZI channel display unavailable; using saved overlay: {error}", file=sys.stderr)
                self._load_fallback_overlay()
            return

        self._load_fallback_overlay()

    def _ensure_channel_cache(self) -> Path:
        """Return a local NPZ cache, creating it with the analysis runtime."""
        cache = self.channel_cache
        if cache is None:
            cache = self.results_path.parent / ".review_cache" / f"{infer_stem(self.results_path)}.npz"
        if cache.is_file():
            self.channel_cache = cache
            return cache

        try:
            extract_channel_cache(
                self.czi_path,
                self.scene,
                self.nuclei_channel,
                self.actin_channel,
                self.mf20_channel,
                cache,
            )
        except ModuleNotFoundError:
            configured = os.environ.get("MYOFUSE_ANALYSIS_PYTHON")
            candidates = [
                Path(configured).expanduser() if configured else None,
                Path(__file__).resolve().parent / ".venv" / "bin" / "python",
            ]
            analysis_python = next((path for path in candidates if path and path.is_file()), None)
            if analysis_python is None:
                raise FileNotFoundError(
                    "No CZI-capable analysis Python was found. Set "
                    "MYOFUSE_ANALYSIS_PYTHON or run setup_mac.sh."
                )
            command = [
                str(analysis_python),
                str(Path(__file__).resolve()),
                "--czi",
                str(self.czi_path),
                "--scene",
                self.scene,
                "--nuclei-channel",
                str(self.nuclei_channel),
                "--actin-channel",
                str(self.actin_channel),
                "--mf20-channel",
                str(self.mf20_channel),
                "--extract-channel-cache",
                str(cache),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        self.channel_cache = cache
        return cache

    def _load_channel_cache(self, path: Path) -> None:
        with np.load(path) as cached:
            for name in ("DAPI", "Actin", "MF20"):
                if name not in cached:
                    raise ValueError(f"Channel cache is missing {name}")
                channel = cached[name].astype(np.uint8)
                if channel.shape != self.labels.shape:
                    raise ValueError(f"Cached {name} channel shape differs from the label mask")
                self.channels[name] = channel

    def _load_fallback_overlay(self) -> None:

        stem = infer_stem(self.results_path)
        overlay_path = find_related_file(
            self.results_path,
            [
                f"{stem}__hybrid_nucleus_aware_overlay.tif",
                f"{stem}_hybrid_nucleus_aware_overlay.tif",
            ],
        )
        if overlay_path is not None:
            overlay = imread(overlay_path)
            if overlay.ndim == 2:
                overlay = np.repeat(overlay[..., None], 3, axis=2)
            if overlay.shape[:2] == self.labels.shape:
                self.fallback_background = overlay[..., :3].astype(np.uint8)
        if self.fallback_background is None:
            self.fallback_background = np.zeros((*self.labels.shape, 3), dtype=np.uint8)

    @property
    def total_automatic(self) -> int:
        return len(self.automatic)

    @property
    def automatic_in(self) -> int:
        return sum(state == IN for state in self.automatic.values())

    @property
    def valid_total(self) -> int:
        return sum(state != EXCLUDE for state in self.current.values())

    @property
    def corrected_in(self) -> int:
        return sum(state == IN for state in self.current.values())

    @property
    def automatic_fi(self) -> float:
        return 100 * self.automatic_in / self.total_automatic if self.total_automatic else float("nan")

    @property
    def corrected_fi(self) -> float:
        return 100 * self.corrected_in / self.valid_total if self.valid_total else float("nan")

    @property
    def active_added(self) -> list[int]:
        return [label_id for label_id in self.added if label_id in self.current]

    @property
    def changed_count(self) -> int:
        changed = sum(self.current[label_id] != state for label_id, state in self.automatic.items())
        return changed + len(self.active_added)

    def _added_box(self, label_id: int):
        x, y, radius = self.added[label_id]
        r = int(np.ceil(radius)) + 1
        height, width = self.labels.shape
        y0, y1 = max(0, int(y) - r), min(height, int(y) + r + 1)
        x0, x1 = max(0, int(x) - r), min(width, int(x) + r + 1)
        return y0, y1, x0, x1

    def _paint_added(self, label_id: int) -> int:
        """Draw an added nucleus as a disc on empty pixels only."""
        x, y, radius = self.added[label_id]
        y0, y1, x0, x1 = self._added_box(label_id)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        window = self.labels[y0:y1, x0:x1]
        disc = ((xx - x) ** 2 + (yy - y) ** 2 <= radius * radius) & (window == 0)
        window[disc] = label_id
        return int(disc.sum())

    def _erase_added(self, label_id: int) -> None:
        y0, y1, x0, x1 = self._added_box(label_id)
        window = self.labels[y0:y1, x0:x1]
        window[window == label_id] = 0

    def _place(self, label_id: int, state: int) -> None:
        """Apply a state during undo/redo, including add/remove of added nuclei."""
        if state == ABSENT:
            self.current.pop(label_id, None)
            self.centres.pop(label_id, None)
            self._erase_added(label_id)
            return
        if label_id not in self.current and label_id in self.added:
            self._paint_added(label_id)
            self.centres[label_id] = self.added[label_id][:2]
        self.current[label_id] = state

    def add_nucleus(self, x: float, y: float, state: int = IN,
                    radius: Optional[float] = None) -> Optional[int]:
        """Add a nucleus that segmentation missed. Returns its new label id."""
        if state not in STATE_NAMES:
            return None
        xi, yi = int(round(x)), int(round(y))
        height, width = self.labels.shape
        if not (0 <= xi < width and 0 <= yi < height) or self.labels[yi, xi] != 0:
            return None
        radius = float(radius) if radius else self.typical_radius
        radius = max(2.0, min(radius, 60.0))
        label_id = max(int(self.labels.max()), max(self.automatic, default=0),
                       max(self.added, default=0)) + 1
        self.added[label_id] = (float(x), float(y), radius)
        if self._paint_added(label_id) == 0:
            del self.added[label_id]
            return None
        self.current[label_id] = state
        self.centres[label_id] = (float(x), float(y))
        self.history.append(Change(label_id=label_id, old_state=ABSENT, new_state=state, action="add nucleus"))
        self.redo_stack.clear()
        return label_id

    def remove_added(self, label_id: int) -> bool:
        """Remove a manually added nucleus. Segmented nuclei cannot be removed, only excluded."""
        if label_id not in self.added or label_id not in self.current:
            return False
        old = self.current[label_id]
        self._place(label_id, ABSENT)
        self.history.append(Change(label_id=label_id, old_state=old, new_state=ABSENT, action="remove added nucleus"))
        self.redo_stack.clear()
        return True

    def set_state(self, label_id: int, state: int, action: str = "set") -> bool:
        if label_id not in self.current or state not in STATE_NAMES:
            return False
        old = self.current[label_id]
        if old == state:
            return False
        change = Change(label_id=label_id, old_state=old, new_state=state, action=action)
        self.current[label_id] = state
        self.history.append(change)
        self.redo_stack.clear()
        return True

    def toggle(self, label_id: int) -> bool:
        old = self.current[label_id]
        new = IN if old != IN else OUT
        return self.set_state(label_id, new, "toggle")

    def reset(self, label_id: int) -> bool:
        if label_id not in self.automatic:
            return False
        return self.set_state(label_id, self.automatic[label_id], "reset")

    def undo(self) -> Optional[int]:
        if not self.history:
            return None
        change = self.history.pop()
        self._place(change.label_id, change.old_state)
        self.redo_stack.append(change)
        return change.label_id

    def redo(self) -> Optional[int]:
        if not self.redo_stack:
            return None
        change = self.redo_stack.pop()
        self._place(change.label_id, change.new_state)
        self.history.append(change)
        return change.label_id

    def label_at(self, x: float, y: float, search_radius: float = 12.0) -> Optional[int]:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < self.labels.shape[0] and 0 <= xi < self.labels.shape[1]:
            label_id = int(self.labels[yi, xi])
            if label_id in self.current:
                return label_id
        nearest = None
        best = search_radius * search_radius
        for label_id, (cx, cy) in self.centres.items():
            distance = (cx - x) ** 2 + (cy - y) ** 2
            if distance <= best:
                nearest, best = label_id, distance
        return nearest

    def ids_in_box(self, x1: float, y1: float, x2: float, y2: float) -> list[int]:
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return [
            label_id
            for label_id, (x, y) in self.centres.items()
            if left <= x <= right and top <= y <= bottom
        ]

    def make_background(self, show_dapi: bool, show_actin: bool, show_mf20: bool,
                        pseudocolor: bool = True) -> np.ndarray:
        if not self.channels:
            return self.fallback_background.copy()
        selected = [name for name, show in (("DAPI", show_dapi), ("Actin", show_actin), ("MF20", show_mf20)) if show]
        if not selected:
            return np.zeros((*self.labels.shape, 3), dtype=np.uint8)
        if len(selected) == 1 and not pseudocolor:
            gray = self.channels[selected[0]]
            return np.repeat(gray[..., None], 3, axis=2)
        rgb = np.zeros((*self.labels.shape, 3), dtype=np.float32)
        if show_mf20:
            rgb[..., 0] += self.channels["MF20"]
        if show_actin:
            rgb[..., 1] += self.channels["Actin"]
        if show_dapi:
            rgb[..., 2] += self.channels["DAPI"]
            rgb[..., 1] += 0.25 * self.channels["DAPI"]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def render_overlay(
        self,
        show_dapi: bool = False,
        show_actin: bool = False,
        show_mf20: bool = True,
        show_nuclei: bool = True,
        show_myotube: bool = True,
        show_in: bool = True,
        show_out: bool = True,
        show_review: bool = True,
        show_ids: bool = False,
        selected_id: Optional[int] = None,
        display_gain: float = 1.0,
        pseudocolor: bool = True,
    ) -> np.ndarray:
        image = self.make_background(show_dapi, show_actin, show_mf20, pseudocolor)
        if display_gain != 1.0:
            image = np.clip(image.astype(np.float32) * display_gain, 0, 255).astype(np.uint8)
        if show_myotube and self.myotube_mask is not None:
            # A one-pixel outline disappears when a 1000-pixel field is fitted
            # into a laptop window. Dilate the display outline only; the mask
            # used for classification and FI calculation is unchanged.
            boundary = binary_dilation(
                find_boundaries(self.myotube_mask, mode="inner"), disk(2)
            )
            image[boundary] = (0, 210, 255)
        if show_nuclei:
            boundary = find_boundaries(self.labels, mode="inner")
            lookup = np.full(self.labels.max() + 1, EXCLUDE, dtype=np.int8)
            for label_id, state in self.current.items():
                lookup[label_id] = state
            state_map = lookup[self.labels]
            # Yellow identifies an automatic review candidate. It remains
            # excluded from FI until the reviewer explicitly sets it to In.
            candidate_lookup = np.zeros(self.labels.max() + 1, dtype=bool)
            active_candidates = [
                label_id for label_id in self.review_candidates
                if self.current.get(label_id) == OUT
            ]
            candidate_lookup[active_candidates] = True
            candidate_map = candidate_lookup[self.labels]
            if show_in:
                image[boundary & (state_map == IN)] = STATE_COLORS[IN]
            if show_out:
                image[boundary & (state_map == OUT) & ~candidate_map] = STATE_COLORS[OUT]
            if show_review:
                image[boundary & candidate_map] = REVIEW_CANDIDATE_COLOR
            image[boundary & (state_map == EXCLUDE)] = STATE_COLORS[EXCLUDE]
            # A small white dot marks nuclei that the reviewer added by hand.
            for label_id in self.active_added:
                x, y = self.centres[label_id]
                xi, yi = int(round(x)), int(round(y))
                image[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2] = (255, 255, 255)
            if selected_id is not None and selected_id in self.current:
                selected = binary_dilation(self.labels == selected_id, disk(2))
                ring = find_boundaries(selected, mode="outer")
                image[ring] = (255, 230, 0)
        if show_ids:
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            font = ImageFont.load_default()
            for label_id, (x, y) in self.centres.items():
                state = self.current[label_id]
                is_review = state == OUT and label_id in self.review_candidates
                if state == IN and not show_in:
                    continue
                if is_review and not show_review:
                    continue
                if state == OUT and not is_review and not show_out:
                    continue
                draw.text((x + 4, y - 8), str(label_id), fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
            image = np.asarray(pil)
        return image

    def export(self, output_dir: Path) -> dict[str, Path]:
        """Export corrected data while preserving the original automatic result."""
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = infer_stem(self.results_path)

        reviewed = self.per_nucleus.copy()
        reviewed["Manually added"] = False
        if self.active_added:
            extra = pd.DataFrame(
                [
                    {
                        "Label ID": label_id,
                        "X": self.added[label_id][0],
                        "Y": self.added[label_id][1],
                        "Area": int((self.labels == label_id).sum()),
                        "Manually added": True,
                    }
                    for label_id in self.active_added
                ]
            )
            if "Area" not in reviewed.columns:
                extra = extra.drop(columns="Area")
            reviewed = pd.concat([reviewed, extra], ignore_index=True)
        reviewed = reviewed.loc[reviewed["Label ID"].astype(int).isin(self.current)].copy()
        reviewed["Automatic class used for review"] = reviewed["Label ID"].astype(int).map(self.automatic)
        reviewed["Manual review state"] = reviewed["Label ID"].astype(int).map(self.current).map(STATE_NAMES)
        reviewed["Corrected Class"] = reviewed["Label ID"].astype(int).map(
            lambda label_id: self.current[label_id] if self.current[label_id] != EXCLUDE else pd.NA
        )
        reviewed["Manually corrected"] = reviewed["Label ID"].astype(int).map(
            lambda label_id: self.current[label_id] != self.automatic.get(label_id)
        )
        reviewed["Correction type"] = reviewed["Label ID"].astype(int).map(
            lambda label_id: (
                "Manually added"
                if label_id not in self.automatic
                else "Excluded"
                if self.current[label_id] == EXCLUDE
                else "False positive: automatic In, corrected Out"
                if self.automatic[label_id] == IN and self.current[label_id] == OUT
                else "False negative: automatic Out, corrected In"
                if self.automatic[label_id] == OUT and self.current[label_id] == IN
                else "Unchanged"
            )
        )

        corrections = reviewed.loc[
            reviewed["Manually corrected"],
            ["Label ID", "X", "Y", "Automatic class used for review", "Manual review state", "Corrected Class", "Manually added"],
        ].copy()
        correction_path = output_dir / f"{stem}__manual_corrections.csv"
        corrections.to_csv(correction_path, index=False)

        audit = pd.DataFrame(
            [
                {
                    "Timestamp": change.timestamp,
                    "Label ID": change.label_id,
                    "Action": change.action,
                    "Previous state": state_name(change.old_state),
                    "New state": state_name(change.new_state),
                }
                for change in self.history
            ],
            columns=["Timestamp", "Label ID", "Action", "Previous state", "New state"],
        )
        summary = pd.DataFrame(
            [
                {
                    "Source results": str(self.results_path),
                    "Source CZI": str(self.czi_path) if self.czi_path else "",
                    "Scene": self.scene,
                    "Automatic total nuclei": self.total_automatic,
                    "Automatic nuclei In": self.automatic_in,
                    "Automatic FI (%)": self.automatic_fi,
                    "Corrected valid nuclei": self.valid_total,
                    "Corrected nuclei In": self.corrected_in,
                    "Excluded nuclei": sum(state == EXCLUDE for state in self.current.values()),
                    "Changed nuclei": self.changed_count,
                    "Manually added nuclei": len(self.active_added),
                    "Manually added nuclei In": sum(self.current[i] == IN for i in self.active_added),
                    "False-positive corrections (In to Out)": sum(
                        self.automatic.get(label_id) == IN and state == OUT
                        for label_id, state in self.current.items()
                    ),
                    "False-negative corrections (Out to In)": sum(
                        self.automatic.get(label_id) == OUT and state == IN
                        for label_id, state in self.current.items()
                    ),
                    "Corrected FI (%)": self.corrected_fi,
                    "Exported at": utc_now(),
                }
            ]
        )
        workbook_path = output_dir / f"{stem}__manually_reviewed_results.xlsx"
        with pd.ExcelWriter(workbook_path) as writer:
            summary.to_excel(writer, index=False, sheet_name="Corrected summary")
            if "Summary" in self.sheets:
                self.sheets["Summary"].to_excel(writer, index=False, sheet_name="Automatic summary")
            if "Parameters" in self.sheets:
                self.sheets["Parameters"].to_excel(writer, index=False, sheet_name="Parameters")
            reviewed.to_excel(writer, index=False, sheet_name="Per nucleus")
            audit.to_excel(writer, index=False, sheet_name="Correction log")

        overlay = self.render_overlay(show_dapi=False, show_actin=False, show_mf20=True, show_ids=False)
        overlay_tif = output_dir / f"{stem}__manually_reviewed_overlay.tif"
        overlay_png = output_dir / f"{stem}__manually_reviewed_preview.png"
        imwrite(overlay_tif, overlay, photometric="rgb")
        self._save_preview(overlay_png, overlay)
        reviewed_labels = output_dir / f"{stem}__reviewed_label_mask.tif"
        imwrite(reviewed_labels, self.labels.astype(np.int32))

        project_path = output_dir / f"{stem}__review_project.json"
        project_data = {
            "format": "MyoFuse-Hybrid manual review v1",
            "results_path": str(self.results_path),
            "label_mask_path": str(self.label_mask_path),
            "czi_path": str(self.czi_path) if self.czi_path else None,
            "channel_cache": str(self.channel_cache) if self.channel_cache else None,
            "myotube_mask_path": str(self.myotube_mask_path) if self.myotube_mask_path else None,
            "scene": self.scene,
            "channels": {
                "nuclei": self.nuclei_channel,
                "actin": self.actin_channel,
                "mf20": self.mf20_channel,
            },
            "states": {str(label_id): state for label_id, state in self.current.items()},
            "added_nuclei": {
                str(label_id): {"x": x, "y": y, "radius": r}
                for label_id, (x, y, r) in self.added.items()
            },
            "history": [change.__dict__ for change in self.history],
            "saved_at": utc_now(),
        }
        project_path.write_text(json.dumps(project_data, indent=2), encoding="utf-8")
        return {
            "workbook": workbook_path,
            "corrections": correction_path,
            "overlay_tif": overlay_tif,
            "preview": overlay_png,
            "project": project_path,
            "reviewed_labels": reviewed_labels,
        }

    def _save_preview(self, path: Path, overlay: np.ndarray) -> None:
        height, width = overlay.shape[:2]
        canvas = Image.new("RGB", (width, height + 54), "white")
        canvas.paste(Image.fromarray(overlay), (0, 54))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        title = (
            f"Automatic FI {self.automatic_fi:.2f}%  |  "
            f"Corrected FI {self.corrected_fi:.2f}%  |  "
            f"Changed {self.changed_count}  |  Excluded "
            f"{sum(state == EXCLUDE for state in self.current.values())}"
        )
        draw.text((12, 8), title, fill="black", font=font)
        legend = [((38, 222, 95), "In"), ((255, 77, 79), "Out"), ((255, 220, 0), "review candidate"), ((180, 180, 180), "Exclude"), ((0, 210, 255), "myotube boundary")]
        x = 12
        for color, text in legend:
            draw.line((x, 36, x + 20, 36), fill=color, width=4)
            draw.text((x + 26, 29), text, fill="black", font=font)
            x += 40 + draw.textlength(text, font=font) + 18
        canvas.save(path)

    @classmethod
    def from_saved_project(cls, path: Path) -> "ReviewProject":
        data = json.loads(path.read_text(encoding="utf-8"))
        channels = data.get("channels", {})
        project = cls(
            results_path=Path(data["results_path"]),
            label_mask_path=Path(data["label_mask_path"]),
            czi_path=Path(data["czi_path"]) if data.get("czi_path") else None,
            scene=data.get("scene"),
            nuclei_channel=int(channels.get("nuclei", 0)),
            actin_channel=int(channels.get("actin", 1)),
            mf20_channel=int(channels.get("mf20", 2)),
            channel_cache=Path(data["channel_cache"]) if data.get("channel_cache") else None,
            myotube_mask_path=Path(data["myotube_mask_path"]) if data.get("myotube_mask_path") else None,
        )
        saved_states = {int(key): int(value) for key, value in data.get("states", {}).items()}
        added = {
            int(key): (float(v["x"]), float(v["y"]), float(v["radius"]))
            for key, v in data.get("added_nuclei", {}).items()
        }
        if set(saved_states) - set(added) != set(project.current):
            raise ValueError("Saved project nucleus IDs differ from the source analysis")
        project.added = added
        for label_id in added:
            if label_id in saved_states:
                project._paint_added(label_id)
                project.centres[label_id] = added[label_id][:2]
        project.current = saved_states
        project.history = [Change(**record) for record in data.get("history", [])]
        return project


def create_gui(args: argparse.Namespace) -> int:
    """Create the Tk interface after all non-GUI helpers are defined."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class ReviewApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("MyoFuse Hybrid Review - New Review")
            self.geometry("1450x900")
            self.minsize(1050, 650)
            self.project: Optional[ReviewProject] = None
            self.selected_id: Optional[int] = None
            self.zoom = 1.0
            self.image_tk: Optional[ImageTk.PhotoImage] = None
            self.image_item: Optional[int] = None
            self.drag_start: Optional[tuple[float, float]] = None
            self.selection_rectangle: Optional[int] = None
            self.selection_moved = False
            self.tool = tk.StringVar(value="Toggle")
            self.show_dapi = tk.BooleanVar(value=False)
            self.show_actin = tk.BooleanVar(value=False)
            self.show_mf20 = tk.BooleanVar(value=True)
            self.show_nuclei = tk.BooleanVar(value=True)
            self.show_myotube = tk.BooleanVar(value=True)
            self.show_ids = tk.BooleanVar(value=False)
            self.status = tk.StringVar(value="Open a hybrid result workbook to begin.")
            self.stats = {name: tk.StringVar(value="-") for name in ("auto", "corrected", "in", "out", "excluded", "changed")}
            self.selected_text = tk.StringVar(value="No nucleus selected")
            self._configure_style()
            self._make_menu()
            self._make_layout()
            self.protocol("WM_DELETE_WINDOW", self.destroy)
            if args.project:
                self.after(50, lambda: self.load_saved_project(Path(args.project)))
            elif args.results and args.label_mask:
                self.after(50, self._load_from_args)

        def _configure_style(self) -> None:
            style = ttk.Style(self)
            try:
                style.theme_use("aqua")
            except tk.TclError:
                pass
            style.configure("Title.TLabel", font=("Helvetica", 17, "bold"))
            style.configure("Metric.TLabel", font=("Helvetica", 16, "bold"))
            style.configure("Muted.TLabel", foreground="#60656f")
            style.configure("Selected.Tool.TButton", font=("Helvetica", 11, "bold"))

        def _make_menu(self) -> None:
            menu = tk.Menu(self)
            file_menu = tk.Menu(menu, tearoff=False)
            file_menu.add_command(label="Open Analysis...", command=self.open_analysis)
            file_menu.add_command(label="Open Review Project...", command=self.open_saved_project)
            file_menu.add_separator()
            file_menu.add_command(label="Save Review / Export...", command=self.export_review, accelerator="⌘S")
            file_menu.add_separator()
            file_menu.add_command(label="Quit", command=self.destroy)
            menu.add_cascade(label="File", menu=file_menu)
            edit_menu = tk.Menu(menu, tearoff=False)
            edit_menu.add_command(label="Undo", command=self.undo, accelerator="⌘Z")
            edit_menu.add_command(label="Redo", command=self.redo, accelerator="⇧⌘Z")
            edit_menu.add_command(label="Reset Selected to Automatic", command=self.reset_selected)
            menu.add_cascade(label="Edit", menu=edit_menu)
            help_menu = tk.Menu(menu, tearoff=False)
            help_menu.add_command(label="Mouse Controls", command=self.show_help)
            menu.add_cascade(label="Help", menu=help_menu)
            self.config(menu=menu)
            self.bind_all("<Command-s>", lambda _: self.export_review())
            self.bind_all("<Control-s>", lambda _: self.export_review())
            self.bind_all("<Command-z>", lambda _: self.undo())
            self.bind_all("<Control-z>", lambda _: self.undo())
            self.bind_all("<Command-Shift-z>", lambda _: self.redo())

        def _make_layout(self) -> None:
            top = ttk.Frame(self, padding=(10, 8))
            top.pack(side="top", fill="x")
            ttk.Button(top, text="Open Analysis", command=self.open_analysis).pack(side="left", padx=(0, 8))
            ttk.Button(top, text="Save & Export", command=self.export_review).pack(side="left", padx=(0, 18))
            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=(0, 14))
            ttk.Label(top, text="Review tool:").pack(side="left", padx=(0, 5))
            for tool in ("Toggle", "Set In", "Set Out", "Exclude", "Reset"):
                ttk.Radiobutton(top, text=tool, value=tool, variable=self.tool).pack(side="left", padx=3)
            ttk.Button(top, text="Undo", command=self.undo).pack(side="right", padx=3)
            ttk.Button(top, text="Redo", command=self.redo).pack(side="right", padx=3)

            # A regular packed frame is used here instead of ttk.Panedwindow.
            # The Aqua/Tk 8.5 build included with some macOS Python 3.9
            # installations can initially assign a zero width to a pane,
            # leaving the image canvas invisible.
            body = ttk.Frame(self)
            body.pack(fill="both", expand=True, padx=8, pady=(0, 5))
            canvas_frame = ttk.Frame(body)
            side = ttk.Frame(body, width=330, padding=(12, 8))
            side.pack(side="right", fill="y")
            side.pack_propagate(False)
            canvas_frame.pack(side="left", fill="both", expand=True)

            self.canvas = tk.Canvas(canvas_frame, bg="#121418", highlightthickness=1, highlightbackground="#333840")
            hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
            vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
            self.canvas.grid(row=0, column=0, sticky="nsew")
            vbar.grid(row=0, column=1, sticky="ns")
            hbar.grid(row=1, column=0, sticky="ew")
            canvas_frame.rowconfigure(0, weight=1)
            canvas_frame.columnconfigure(0, weight=1)
            self.canvas.bind("<MouseWheel>", self.on_zoom)
            self.canvas.bind("<Button-4>", lambda event: self.on_zoom(event, 1))
            self.canvas.bind("<Button-5>", lambda event: self.on_zoom(event, -1))
            self.canvas.bind("<ButtonPress-2>", self.start_pan)
            self.canvas.bind("<B2-Motion>", self.pan)
            self.canvas.bind("<ButtonPress-1>", self.start_selection)
            self.canvas.bind("<B1-Motion>", self.move_selection)
            self.canvas.bind("<ButtonRelease-1>", self.finish_selection)
            self.canvas.bind("<ButtonPress-3>", self.start_selection)
            self.canvas.bind("<B3-Motion>", self.move_selection)
            self.canvas.bind("<ButtonRelease-3>", self.finish_exclusion_selection)

            ttk.Label(side, text="MyoFuse Hybrid Review", style="Title.TLabel").pack(anchor="w")
            self.file_label = ttk.Label(side, text="No analysis loaded", style="Muted.TLabel", wraplength=300)
            self.file_label.pack(anchor="w", fill="x", pady=(3, 12))

            metrics = ttk.LabelFrame(side, text="Fusion index", padding=10)
            metrics.pack(fill="x", pady=(0, 10))
            ttk.Label(metrics, text="Automatic").grid(row=0, column=0, sticky="w")
            ttk.Label(metrics, textvariable=self.stats["auto"], style="Metric.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 8))
            ttk.Label(metrics, text="After manual review").grid(row=0, column=1, sticky="w", padx=(30, 0))
            ttk.Label(metrics, textvariable=self.stats["corrected"], style="Metric.TLabel").grid(row=1, column=1, sticky="w", padx=(30, 0), pady=(0, 8))
            for row, (label, key) in enumerate((("In", "in"), ("Out", "out"), ("Excluded", "excluded"), ("Changed", "changed")), start=2):
                ttk.Label(metrics, text=label).grid(row=row, column=0, sticky="w", pady=2)
                ttk.Label(metrics, textvariable=self.stats[key]).grid(row=row, column=1, sticky="e", pady=2)
            metrics.columnconfigure(1, weight=1)

            display = ttk.LabelFrame(side, text="Display", padding=10)
            display.pack(fill="x", pady=(0, 10))
            for text, variable in (("DAPI", self.show_dapi), ("Actin", self.show_actin), ("MF20", self.show_mf20), ("Nuclei", self.show_nuclei), ("Myotube boundary", self.show_myotube), ("Label IDs", self.show_ids)):
                ttk.Checkbutton(display, text=text, variable=variable, command=self.refresh).pack(anchor="w", pady=2)

            legend = ttk.LabelFrame(side, text="Legend", padding=10)
            legend.pack(fill="x", pady=(0, 10))
            for color, text in (("#26de5f", "In"), ("#ff4d4f", "Out"), ("#ffdc00", "Review candidate"), ("#b4b4b4", "Excluded"), ("#00d2ff", "Myotube boundary"), ("#ffe600", "Selected")):
                row = ttk.Frame(legend)
                row.pack(fill="x", pady=2)
                swatch = tk.Canvas(row, width=22, height=10, highlightthickness=0)
                swatch.create_rectangle(1, 1, 21, 9, fill=color, outline=color)
                swatch.pack(side="left", padx=(0, 8))
                ttk.Label(row, text=text).pack(side="left")

            selected = ttk.LabelFrame(side, text="Selected nucleus", padding=10)
            selected.pack(fill="x", pady=(0, 10))
            ttk.Label(selected, textvariable=self.selected_text, wraplength=285).pack(anchor="w")
            ttk.Button(selected, text="Reset to automatic", command=self.reset_selected).pack(anchor="w", pady=(8, 0))

            instructions = ttk.LabelFrame(side, text="Mouse controls", padding=10)
            instructions.pack(fill="x")
            ttk.Label(
                instructions,
                text=(
                    "Left click: apply selected tool\n"
                    "Left drag: edit nuclei in a box\n"
                    "Right click/drag: exclude\n"
                    "Mouse wheel: zoom\n"
                    "Middle drag: pan"
                ),
                justify="left",
            ).pack(anchor="w")

            status = ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 4))
            status.pack(side="bottom", fill="x")

        def _load_from_args(self) -> None:
            self.load_project(
                results=Path(args.results),
                labels=Path(args.label_mask),
                czi=Path(args.czi) if args.czi else None,
                scene=args.scene,
                channel_cache=Path(args.channel_cache) if args.channel_cache else None,
            )

        def open_analysis(self) -> None:
            results = filedialog.askopenfilename(title="Select hybrid results workbook", filetypes=[("Excel workbook", "*.xlsx")])
            if not results:
                return
            results_path = Path(results)
            stem = infer_stem(results_path)
            label_path = find_related_file(results_path, [f"{stem}_mask.tif", f"{stem}__mask.tif"])
            if label_path is None:
                selected = filedialog.askopenfilename(title="Select the MyoFuse label mask", filetypes=[("TIFF label mask", "*.tif *.tiff")])
                if not selected:
                    return
                label_path = Path(selected)
            czi = filedialog.askopenfilename(title="Select the original CZI (Cancel to use saved overlay)", filetypes=[("Zeiss CZI", "*.czi")])
            self.load_project(results_path, label_path, Path(czi) if czi else None, None)

        def load_project(
            self,
            results: Path,
            labels: Path,
            czi: Optional[Path],
            scene: Optional[str],
            channel_cache: Optional[Path] = None,
        ) -> None:
            self.status.set("Loading analysis...")
            self.update_idletasks()
            try:
                self.project = ReviewProject(
                    results_path=results,
                    label_mask_path=labels,
                    czi_path=czi,
                    scene=scene,
                    channel_cache=channel_cache,
                )
            except Exception as error:
                self.project = None
                messagebox.showerror("Could not open analysis", str(error))
                self.status.set("Analysis was not loaded.")
                return
            self.selected_id = None
            self.zoom = 1.0
            self.title(f"MyoFuse Hybrid Review - {results.name}")
            self.file_label.configure(text=f"{results.name}\nScene: {self.project.scene}")
            self.update_stats()
            self.refresh(fit=True)
            self.status.set("Loaded. Left click a nucleus to review it.")

        def open_saved_project(self) -> None:
            path = filedialog.askopenfilename(title="Open review project", filetypes=[("Review project", "*.json")])
            if path:
                self.load_saved_project(Path(path))

        def load_saved_project(self, path: Path) -> None:
            try:
                self.project = ReviewProject.from_saved_project(path)
            except Exception as error:
                messagebox.showerror("Could not open review project", str(error))
                return
            self.selected_id = None
            self.zoom = 1.0
            self.title(f"MyoFuse Hybrid Review - {path.name}")
            self.file_label.configure(text=f"{self.project.results_path.name}\nScene: {self.project.scene}")
            self.update_stats()
            self.refresh(fit=True)
            self.status.set(f"Review project loaded: {path.name}")

        def refresh(self, fit: bool = False) -> None:
            if self.project is None:
                return
            array = self.project.render_overlay(
                self.show_dapi.get(),
                self.show_actin.get(),
                self.show_mf20.get(),
                self.show_nuclei.get(),
                self.show_myotube.get(),
                self.show_ids.get(),
                self.selected_id,
            )
            if fit:
                self.update_idletasks()
                available_w = max(self.canvas.winfo_width() - 30, 200)
                available_h = max(self.canvas.winfo_height() - 30, 200)
                self.zoom = min(1.0, available_w / array.shape[1], available_h / array.shape[0])
            width = max(1, int(array.shape[1] * self.zoom))
            height = max(1, int(array.shape[0] * self.zoom))
            pil = Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(pil)
            if self.image_item is None:
                self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            else:
                self.canvas.itemconfigure(self.image_item, image=self.image_tk)
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self.update_selected_text()

        def update_stats(self) -> None:
            if self.project is None:
                return
            p = self.project
            excluded = sum(state == EXCLUDE for state in p.current.values())
            out = sum(state == OUT for state in p.current.values())
            self.stats["auto"].set(f"{p.automatic_fi:.2f}%")
            self.stats["corrected"].set(f"{p.corrected_fi:.2f}%")
            self.stats["in"].set(str(p.corrected_in))
            self.stats["out"].set(str(out))
            self.stats["excluded"].set(str(excluded))
            self.stats["changed"].set(str(p.changed_count))

        def update_selected_text(self) -> None:
            if self.project is None or self.selected_id is None:
                self.selected_text.set("No nucleus selected")
                return
            label_id = self.selected_id
            automatic = STATE_NAMES.get(self.project.automatic.get(label_id), "Manually added")
            current = STATE_NAMES[self.project.current[label_id]]
            x, y = self.project.centres[label_id]
            self.selected_text.set(f"Label ID: {label_id}\nAutomatic: {automatic}\nCurrent: {current}\nPosition: ({x:.1f}, {y:.1f})")

        def canvas_coordinates(self, event) -> tuple[float, float]:
            return self.canvas.canvasx(event.x) / self.zoom, self.canvas.canvasy(event.y) / self.zoom

        def apply_tool(self, label_ids: list[int], force_exclude: bool = False) -> None:
            if self.project is None or not label_ids:
                return
            tool = "Exclude" if force_exclude else self.tool.get()
            changed = 0
            for label_id in label_ids:
                if tool == "Toggle":
                    changed += int(self.project.toggle(label_id))
                elif tool == "Set In":
                    changed += int(self.project.set_state(label_id, IN, "set In"))
                elif tool == "Set Out":
                    changed += int(self.project.set_state(label_id, OUT, "set Out"))
                elif tool == "Exclude":
                    changed += int(self.project.set_state(label_id, EXCLUDE, "exclude"))
                elif tool == "Reset":
                    changed += int(self.project.reset(label_id))
            self.selected_id = label_ids[-1]
            if changed:
                self.update_stats()
                self.refresh()
                self.status.set(f"Updated {changed} nucleus/nuclei. Changes are not exported yet.")
            else:
                self.refresh()

        def start_selection(self, event) -> None:
            if self.project is None:
                return
            self.drag_start = self.canvas_coordinates(event)
            self.selection_moved = False

        def move_selection(self, event) -> None:
            if self.drag_start is None:
                return
            x1, y1 = self.drag_start
            x2, y2 = self.canvas_coordinates(event)
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 < 16:
                return
            self.selection_moved = True
            coords = (x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom)
            if self.selection_rectangle is None:
                self.selection_rectangle = self.canvas.create_rectangle(*coords, outline="#ffe600", width=2)
            else:
                self.canvas.coords(self.selection_rectangle, *coords)

        def finish_selection(self, event) -> None:
            if self.project is None or self.drag_start is None:
                return
            end = self.canvas_coordinates(event)
            if self.selection_moved:
                ids = self.project.ids_in_box(*self.drag_start, *end)
            else:
                label_id = self.project.label_at(*end)
                ids = [label_id] if label_id is not None else []
            if self.selection_rectangle is not None:
                self.canvas.delete(self.selection_rectangle)
            self.selection_rectangle = None
            self.drag_start = None
            self.selection_moved = False
            self.apply_tool(ids)

        def finish_exclusion_selection(self, event) -> None:
            if self.project is None or self.drag_start is None:
                return
            end = self.canvas_coordinates(event)
            if self.selection_moved:
                ids = self.project.ids_in_box(*self.drag_start, *end)
            else:
                label_id = self.project.label_at(*end)
                ids = [label_id] if label_id is not None else []
            if self.selection_rectangle is not None:
                self.canvas.delete(self.selection_rectangle)
            self.selection_rectangle = None
            self.drag_start = None
            self.selection_moved = False
            self.apply_tool(ids, force_exclude=True)

        def start_pan(self, event) -> None:
            self.canvas.scan_mark(event.x, event.y)

        def pan(self, event) -> None:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

        def on_zoom(self, event, direction: Optional[int] = None) -> None:
            if self.project is None:
                return
            if direction is None:
                direction = 1 if event.delta > 0 else -1
            old_zoom = self.zoom
            new_zoom = min(5.0, max(0.15, old_zoom * (1.18 if direction > 0 else 1 / 1.18)))
            if abs(new_zoom - old_zoom) < 1e-6:
                return
            x_image, y_image = self.canvas_coordinates(event)
            self.zoom = new_zoom
            self.refresh()
            width = self.project.labels.shape[1] * self.zoom
            height = self.project.labels.shape[0] * self.zoom
            target_x = x_image * self.zoom - event.x
            target_y = y_image * self.zoom - event.y
            if width > self.canvas.winfo_width():
                self.canvas.xview_moveto(max(0, target_x) / width)
            if height > self.canvas.winfo_height():
                self.canvas.yview_moveto(max(0, target_y) / height)

        def reset_selected(self) -> None:
            if self.project is not None and self.selected_id is not None:
                self.apply_tool([self.selected_id]) if self.tool.get() == "Reset" else self._reset_id(self.selected_id)

        def _reset_id(self, label_id: int) -> None:
            if self.project and self.project.reset(label_id):
                self.update_stats()
                self.refresh()
                self.status.set(f"Label {label_id} reset to its automatic class.")

        def undo(self) -> None:
            if self.project is None:
                return
            label_id = self.project.undo()
            if label_id is not None:
                self.selected_id = label_id
                self.update_stats()
                self.refresh()
                self.status.set(f"Undid the last change to label {label_id}.")

        def redo(self) -> None:
            if self.project is None:
                return
            label_id = self.project.redo()
            if label_id is not None:
                self.selected_id = label_id
                self.update_stats()
                self.refresh()
                self.status.set(f"Redid the change to label {label_id}.")

        def export_review(self) -> None:
            if self.project is None:
                messagebox.showinfo("Nothing to export", "Open an analysis first.")
                return
            default = self.project.results_path.parent / "manual_review"
            directory = filedialog.askdirectory(title="Choose export folder", initialdir=str(default.parent))
            if not directory:
                return
            try:
                paths = self.project.export(Path(directory))
            except Exception as error:
                messagebox.showerror("Export failed", str(error))
                return
            self.status.set(f"Review exported to {directory}")
            messagebox.showinfo(
                "Review exported",
                "Saved corrected workbook, correction log, overlays, and review project.\n\n"
                f"Workbook: {paths['workbook'].name}",
            )

        def show_help(self) -> None:
            messagebox.showinfo(
                "Manual review controls",
                "Choose a review tool in the top toolbar.\n\n"
                "Left click applies the tool to one nucleus.\n"
                "Left drag applies it to nuclei inside the box.\n"
                "Right click excludes one nucleus.\n"
                "Use the mouse wheel to zoom and the middle button to pan.\n\n"
                "Exclude removes a segmentation object from the corrected FI denominator. "
                "The automatic result remains unchanged in every export.",
            )

    app = ReviewApp()
    app.mainloop()
    return 0


def extract_channel_cache(
    czi_path: Path,
    scene: str,
    nuclei_channel: int,
    actin_channel: int,
    mf20_channel: int,
    output: Path,
) -> Path:
    """Extract display channels with the CZI-capable analysis environment."""
    from czi_tiles import open_analysis_scene, read_channel

    czi_path = czi_path.expanduser().resolve()
    output = output.expanduser().resolve()
    image, tile_index = open_analysis_scene(czi_path, scene)
    if image.dims.T != 1 or image.dims.Z != 1:
        raise ValueError("The review app currently expects one T and one Z plane")
    requested = {
        "DAPI": nuclei_channel,
        "Actin": actin_channel,
        "MF20": mf20_channel,
    }
    channels: dict[str, np.ndarray] = {}
    for name, index in requested.items():
        if not 0 <= index < image.dims.C:
            raise ValueError(f"{name} channel {index} is outside 0..{image.dims.C - 1}")
        channels[name] = scale_channel(read_channel(image, index, tile_index))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **channels)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Hybrid result workbook")
    parser.add_argument("--label-mask", type=Path, help="MyoFuse integer label mask TIFF")
    parser.add_argument("--czi", type=Path, help="Original CZI for channel display")
    parser.add_argument("--scene", help="CZI scene, for example Image:0")
    parser.add_argument("--channel-cache", type=Path, help="Pre-extracted DAPI/Actin/MF20 NPZ cache")
    parser.add_argument("--nuclei-channel", type=int, default=0)
    parser.add_argument("--actin-channel", type=int, default=1)
    parser.add_argument("--mf20-channel", type=int, default=2)
    parser.add_argument("--extract-channel-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--project", type=Path, help="Previously saved review project JSON")
    parser.add_argument(
        "--headless-export",
        type=Path,
        help="Load --results/--label-mask and export without opening the interface (test/helper mode)",
    )
    args = parser.parse_args()
    if not args.extract_channel_cache and bool(args.results) != bool(args.label_mask):
        parser.error("--results and --label-mask must be supplied together")
    return args


def main() -> int:
    args = parse_args()
    if args.extract_channel_cache:
        if not args.czi:
            raise ValueError("--extract-channel-cache requires --czi")
        path = extract_channel_cache(
            args.czi,
            args.scene or "Image:0",
            args.nuclei_channel,
            args.actin_channel,
            args.mf20_channel,
            args.extract_channel_cache,
        )
        print(path)
        return 0
    if args.headless_export:
        if not args.results or not args.label_mask:
            raise ValueError("--headless-export requires --results and --label-mask")
        project = ReviewProject(
            results_path=args.results,
            label_mask_path=args.label_mask,
            czi_path=args.czi,
            scene=args.scene,
            nuclei_channel=args.nuclei_channel,
            actin_channel=args.actin_channel,
            mf20_channel=args.mf20_channel,
            channel_cache=args.channel_cache,
        )
        paths = project.export(args.headless_export)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    return create_gui(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
