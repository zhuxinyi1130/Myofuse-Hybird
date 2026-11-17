#!/usr/bin/env python3
"""Run the published MyoFuse models on multiscene CZI files on macOS.

The segmentation and classification steps follow Ben Lair et al.'s MyoFuse
notebook. This adapter reads every CZI scene (including mosaics) separately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from aicspylibczi import CziFile
from aicsimageio import AICSImage
from cellpose import models
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from tifffile import imwrite

from czi_tiles import open_analysis_scene, read_channel, single_scene_mosaic_tiles


HERE = Path(__file__).resolve().parent
SEG_MODEL = HERE / "models" / "MyoNuclei"
CLASS_MODEL = HERE / "models" / "Svetlana" / "MyoFuse.pth"
CONFIG = HERE / "models" / "Svetlana" / "Config.json"


def normalize_image(image: np.ndarray, percentile_max: float = 99.7) -> np.ndarray:
    """Normalize MF20 while retaining the published MyoFuse knee method.

    A rare saturated pixel can make the fixed-bin histogram very wide and
    move the detected knee above the robust upper percentile. In that case,
    use the first percentile as the background estimate. This keeps the
    normal MyoFuse path unchanged and prevents a bright outlier from aborting
    an otherwise valid field.
    """
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    hist, edges = np.histogram(nonzero, bins=512)
    cdf = np.cumsum(hist) / nonzero.size
    second = np.diff(np.diff(cdf))
    knee = np.argmax(np.convolve(second, np.ones(2) / 2, mode="same")) + 1
    low = edges[knee]
    high = np.percentile(nonzero, percentile_max)
    if high <= low:
        fallback_low = np.percentile(nonzero, 1.0)
        if high <= fallback_low:
            raise ValueError("MF20 channel has insufficient intensity range")
        warnings.warn(
            "MF20 histogram knee was distorted by bright outliers; "
            "using the 1st percentile as the normalization baseline.",
            RuntimeWarning,
        )
        low = fallback_low
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


def centered_patch(array: np.ndarray, row: float, col: float,
                   half_patch_size: int) -> np.ndarray:
    """Extract the same zero-padded, centroid-centered patch used by MyoFuse."""
    pad = half_patch_size + 1
    padding = ((pad, pad), (pad, pad)) + ((0, 0),) * (array.ndim - 2)
    padded = np.pad(array, padding)
    center_row = int(row) + pad
    center_col = int(col) + pad
    h = half_patch_size
    return padded[
        center_row - h:center_row + h,
        center_col - h:center_col + h,
        ...,
    ].copy()


def patch_padding_fraction(shape: tuple[int, int], row: float, col: float,
                           half_patch_size: int) -> float:
    """Return the fraction of a centered patch falling outside the image."""
    center_row, center_col = int(row), int(col)
    h = half_patch_size
    valid_rows = max(0, min(shape[0], center_row + h) - max(0, center_row - h))
    valid_cols = max(0, min(shape[1], center_col + h) - max(0, center_col - h))
    return 1.0 - (valid_rows * valid_cols) / float((2 * h) ** 2)


def hole_metrics(raw: np.ndarray, normalized: np.ndarray, labels: np.ndarray,
                 label_id: int, ring_inner: int,
                 ring_outer: int,
                 label_slice: tuple[slice, slice] | None = None) -> dict[str, float]:
    """Measure MF20 in a nucleus and a nearby ring excluding other nuclei."""
    if label_slice is None:
        coordinates = np.argwhere(labels == label_id)
        if not len(coordinates):
            raise ValueError(f"Missing nucleus label {label_id}")
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
    else:
        y0, y1 = label_slice[0].start, label_slice[0].stop
        x0, x1 = label_slice[1].start, label_slice[1].stop
    y0 = max(0, y0 - ring_outer)
    y1 = min(labels.shape[0], y1 + ring_outer)
    x0 = max(0, x0 - ring_outer)
    x1 = min(labels.shape[1], x1 + ring_outer)
    local_labels = labels[y0:y1, x0:x1]
    nucleus = local_labels == label_id
    inner = binary_dilation(nucleus, disk(ring_inner))
    outer = binary_dilation(nucleus, disk(ring_outer))
    ring = outer & ~inner & (local_labels == 0)

    def means(image: np.ndarray) -> tuple[float, float, float]:
        nucleus_mean = float(image[nucleus].mean()) if nucleus.any() else float("nan")
        ring_mean = float(image[ring].mean()) if ring.any() else float("nan")
        contrast = ((ring_mean - nucleus_mean) / (ring_mean + 1e-12)
                    if np.isfinite(ring_mean) else float("nan"))
        return nucleus_mean, ring_mean, contrast

    raw_nucleus, raw_ring, raw_contrast = means(
        raw[y0:y1, x0:x1].astype(np.float32)
    )
    norm_nucleus, norm_ring, norm_contrast = means(
        normalized[y0:y1, x0:x1]
    )
    return {
        "MF20 Nucleus Mean (Raw)": raw_nucleus,
        "MF20 Ring Mean (Raw)": raw_ring,
        "Hole Contrast (Raw)": raw_contrast,
        "MF20 Nucleus Mean (Normalized)": norm_nucleus,
        "MF20 Ring Mean (Normalized)": norm_ring,
        "Hole Contrast (Normalized)": norm_contrast,
    }


def save_patch_qc(path: Path, raw: np.ndarray, normalized: np.ndarray,
                  labels: np.ndarray, ids: np.ndarray,
                  predictions: np.ndarray, probabilities: np.ndarray,
                  final_predictions: np.ndarray, centroids: list[tuple[float, float]],
                  selected_ids: set[int], half_patch_size: int,
                  ring_inner: int, ring_outer: int) -> None:
    """Save a contact sheet showing every preprocessing stage for selected nuclei."""
    selected = [i for i, label_id in enumerate(ids)
                if int(label_id) in selected_ids]
    if not selected:
        return

    panel = 2 * half_patch_size
    gap = 8
    header = 54
    row_text = 48
    width = 4 * panel + 5 * gap
    height = header + len(selected) * (panel + row_text + gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    headings = ["Raw MF20", "Global normalized", "Patch / patch max",
                "Model input + nucleus/ring"]
    for column, heading in enumerate(headings):
        draw.text((gap + column * (panel + gap), 8), heading,
                  fill="black", font=font)
    draw.text(
        (gap, 27),
        f"Cyan=nucleus boundary; yellow={ring_inner}-{ring_outer}px ring boundary",
        fill="black", font=font,
    )

    scene_high = float(np.percentile(raw[raw > 0], 99.7)) if np.any(raw > 0) else 1.0
    scene_high = max(scene_high, 1.0)
    for row_number, index in enumerate(selected):
        label_id = int(ids[index])
        center_row, center_col = centroids[index]
        raw_patch = centered_patch(raw, center_row, center_col, half_patch_size)
        global_patch = centered_patch(
            normalized, center_row, center_col, half_patch_size
        )
        peak = float(global_patch.max())
        model_patch = global_patch / peak if peak > 0 else global_patch
        mask_patch = centered_patch(
            (labels == label_id).astype(np.uint8), center_row, center_col,
            half_patch_size,
        ).astype(bool)
        inner = binary_dilation(mask_patch, disk(ring_inner))
        outer = binary_dilation(mask_patch, disk(ring_outer))
        ring = outer & ~inner

        displays = [
            np.clip(raw_patch.astype(np.float32) / scene_high, 0, 1),
            global_patch,
            model_patch,
            model_patch,
        ]
        y = header + row_number * (panel + row_text + gap)
        for column, display in enumerate(displays):
            pixels = (np.clip(display, 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(pixels[..., None], 3, axis=2)
            if column == 3:
                rgb[find_boundaries(ring, mode="inner")] = (255, 220, 0)
                rgb[find_boundaries(mask_patch, mode="inner")] = (0, 220, 255)
            image = Image.fromarray(rgb)
            sheet.paste(image, (gap + column * (panel + gap), y))

        metrics = hole_metrics(
            raw, normalized, labels, label_id, ring_inner, ring_outer
        )
        original_class = "In" if predictions[index] == 1 else "Out"
        final_class = "In" if final_predictions[index] == 1 else "Out"
        line1 = (f"ID {label_id}  model={original_class}  p(In)={probabilities[index]:.4f}  "
                 f"final={final_class}  patch peak={peak:.4f}")
        line2 = (f"raw nucleus={metrics['MF20 Nucleus Mean (Raw)']:.1f}  "
                 f"ring={metrics['MF20 Ring Mean (Raw)']:.1f}  "
                 f"hole contrast={metrics['Hole Contrast (Raw)']:.3f}")
        draw.text((gap, y + panel + 4), line1, fill="black", font=font)
        draw.text((gap, y + panel + 20), line2, fill="black", font=font)

    sheet.save(path)


def classify_nuclei(
    image: np.ndarray,
    labels: np.ndarray,
    model: torch.nn.Module,
    half_patch_size: int,
    batch_size: int,
    dilate_mask: bool,
    dilation_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify nuclei with the same four-channel patches as MyoFuse."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        empty = np.array([], dtype=np.uint8)
        return ids, empty, empty.astype(np.float32)

    centroids = ndimage.center_of_mass(
        np.ones_like(labels, dtype=np.float32), labels, ids
    )
    pad = half_patch_size + 1
    rgb = np.repeat(image[..., None], 3, axis=2)
    padded_image = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
    padded_labels = np.pad(labels.astype(np.int32), pad)
    predictions = np.empty(len(ids), dtype=np.uint8)
    probabilities = np.empty(len(ids), dtype=np.float32)

    # The released Config.json disables dilation, but retain its behavior.
    if dilate_mask:
        from skimage.morphology import dilation, disk

        footprint = disk(dilation_size)

    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            patches = []
            for i in range(start, end):
                row, col = (int(x) + pad for x in centroids[i])
                h = half_patch_size
                patch = padded_image[row - h : row + h, col - h : col + h].copy()
                mask = (
                    padded_labels[row - h : row + h, col - h : col + h] == ids[i]
                ).astype(np.float32)
                peak = patch.max()
                if peak > 0:
                    patch /= peak
                if dilate_mask:
                    mask = dilation(mask, footprint)
                    patch *= mask[..., None]
                four_channel = np.empty((4, 2 * h, 2 * h), dtype=np.float32)
                four_channel[:3] = patch.transpose(2, 0, 1)
                four_channel[3] = mask
                patches.append(four_channel)
            batch = torch.from_numpy(np.stack(patches))
            output = model(batch)
            predictions[start:end] = output.argmax(dim=1).numpy()
            probabilities[start:end] = output[:, 1].numpy()
    return ids, predictions, probabilities


def save_overlay(path: Path, image: np.ndarray, labels: np.ndarray,
                 ids: np.ndarray, predictions: np.ndarray,
                 highlighted_ids: set[int] | None = None,
                 highlight_color: tuple[int, int, int] = (255, 220, 0)) -> None:
    base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    overlay = np.repeat(base[..., None], 3, axis=2)
    if ids.size:
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
        lookup[ids] = predictions + 1
        boundary = find_boundaries(labels, mode="inner")
        classified = lookup[labels]
        overlay[boundary & (classified == 1)] = (255, 0, 0)
        overlay[boundary & (classified == 2)] = (0, 255, 0)
        if highlighted_ids:
            highlighted = np.isin(labels, list(highlighted_ids))
            overlay[boundary & highlighted] = highlight_color
    imwrite(path, overlay, photometric="rgb")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def resolve_channel(names: list[str], requested: int | None,
                    expected_name: str, role: str) -> int:
    """Use an explicit index, otherwise find a channel by metadata name."""
    if requested is not None:
        if requested < 0 or requested >= len(names):
            raise ValueError(
                f"{role} channel {requested} is out of range; available: {names}"
            )
        return requested
    matches = [i for i, name in enumerate(names)
               if name.casefold() == expected_name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Cannot automatically identify {role} channel {expected_name!r}; "
        f"available channels: {names}. Specify --{role}-channel."
    )


def overview_scenes(file: Path, scene_names: tuple[str, ...]) -> set[str]:
    """Identify a large stitched scene covering the centers of several fields."""
    if len(scene_names) < 3:
        return set()
    try:
        czi = CziFile(str(file))
        boxes = [czi.get_mosaic_scene_bounding_box(i)
                 for i in range(len(scene_names))]
    except Exception:
        # Some CZI acquisitions do not have mosaic scene coordinates.
        return set()
    median_area = float(np.median([box.w * box.h for box in boxes]))
    overview = set()
    for name, candidate in zip(scene_names, boxes):
        if candidate.w * candidate.h <= 4 * median_area:
            continue
        covered = sum(
            candidate.x <= other.x + other.w / 2 < candidate.x + candidate.w
            and candidate.y <= other.y + other.h / 2 < candidate.y + candidate.h
            for other in boxes if other is not candidate
        )
        if covered >= 2:
            overview.add(name)
    return overview


def run(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    if source.is_dir():
        files = sorted(source.glob("*.czi"))
    elif source.suffix.lower() == ".czi":
        files = [source]
    else:
        raise ValueError("Input must be a .czi file or a folder of .czi files")
    if not files:
        raise ValueError(f"No .czi files found at {source}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    corrections = pd.DataFrame()
    if args.corrections:
        correction_path = args.corrections.expanduser().resolve()
        if correction_path.suffix.lower() == ".csv":
            corrections = pd.read_csv(correction_path)
        else:
            corrections = pd.read_excel(correction_path)
        required = {"File", "Scene", "Label ID", "Corrected Class"}
        missing = required.difference(corrections.columns)
        if missing:
            raise ValueError(
                "Corrections file is missing columns: " + ", ".join(sorted(missing))
            )
        if not corrections["Corrected Class"].isin([0, 1]).all():
            raise ValueError("Corrected Class must contain only 0 or 1")

    with CONFIG.open() as handle:
        config = json.load(handle)
    dilation_config = config["options"]["dilation"]
    dilate_mask = json.loads(dilation_config["dilate_mask"].lower())
    dilation_size = int(dilation_config["str_element_size"])

    torch.set_num_threads(args.threads)
    segmenter = models.CellposeModel(gpu=False, pretrained_model=str(SEG_MODEL))
    # The authors' .pth file contains a pickled Svetlana model class.
    classifier = torch.load(CLASS_MODEL, map_location="cpu", weights_only=False)["model"]
    classifier.eval()
    classifier.softmax.dim = 1
    rows = []
    nucleus_rows = []

    for file in files:
        stitched_image = AICSImage(file, reconstruct_mosaic=True)
        mosaic_tiles = single_scene_mosaic_tiles(file)
        if mosaic_tiles is not None:
            native_scene, tile_scenes = mosaic_tiles
            available_scenes = tuple(tile_scenes) + (native_scene,)
            default_scenes = tuple(tile_scenes)
            overviews = {native_scene}
            print(
                f"{file.name}: detected {len(tile_scenes)} mosaic fields; "
                f"using {', '.join(tile_scenes)} separately",
                flush=True,
            )
        else:
            available_scenes = tuple(stitched_image.scenes)
            overviews = overview_scenes(file, stitched_image.scenes)
            default_scenes = tuple(
                scene for scene in available_scenes if scene not in overviews
            )
        if overviews:
            print(f"{file.name}: overview scenes {', '.join(sorted(overviews))} "
                  "overlap smaller fields; do not pool both", flush=True)
        selected_scenes = tuple(args.scene) if args.scene else default_scenes
        unknown_scenes = set(selected_scenes) - set(available_scenes)
        if unknown_scenes:
            raise ValueError(
                f"{file.name}: scenes not found: {sorted(unknown_scenes)}; "
                f"available: {available_scenes}"
            )
        for scene in selected_scenes:
            start = time.time()
            image, tile_index = open_analysis_scene(file, scene)
            channel_names = list(image.channel_names)
            nuclei_channel = resolve_channel(
                channel_names, args.nuclei_channel, "DAPI", "nuclei"
            )
            myotube_channel = resolve_channel(
                channel_names, args.myotube_channel, "AF647", "myotube"
            )
            if image.dims.T != 1 or image.dims.Z != 1:
                raise ValueError(f"{file.name}/{scene}: T/Z stacks need explicit selection")
            dapi = read_channel(image, nuclei_channel, tile_index)
            myhc = read_channel(image, myotube_channel, tile_index)
            if dapi.shape != myhc.shape:
                raise ValueError(f"{file.name}/{scene}: channel shapes differ")
            stem = safe_name(file.stem + "__" + scene)
            print(
                f"Processing {file.name}/{scene}: {dapi.shape}; "
                f"nuclei C{nuclei_channel}={channel_names[nuclei_channel]}, "
                f"myotube C{myotube_channel}={channel_names[myotube_channel]}",
                flush=True,
            )

            labels, _, _ = segmenter.eval(
                dapi, diameter=args.diameter, channels=[0, 0], normalize=True
            )
            normalized = normalize_image(myhc, args.percentile)
            ids, predictions, probabilities = classify_nuclei(
                normalized, labels, classifier, args.half_patch_size,
                args.batch_size, dilate_mask, dilation_size
            )
            original_predictions = predictions.copy()
            final_predictions = predictions.copy()
            corrected_ids: set[int] = set()
            if not corrections.empty:
                selected = corrections[
                    (corrections["File"] == file.name)
                    & (corrections["Scene"] == scene)
                ]
                index_by_id = {int(label_id): i for i, label_id in enumerate(ids)}
                for _, correction in selected.iterrows():
                    label_id = int(correction["Label ID"])
                    if label_id not in index_by_id:
                        raise ValueError(
                            f"{file.name}/{scene}: correction refers to missing "
                            f"label {label_id}"
                        )
                    final_predictions[index_by_id[label_id]] = int(
                        correction["Corrected Class"]
                    )
                    corrected_ids.add(label_id)

            centroids = ndimage.center_of_mass(
                np.ones_like(labels, dtype=np.float32), labels, ids
            ) if len(ids) else []
            local_means = []
            for row, col in centroids:
                r0, r1 = max(0, int(row) - 40), min(normalized.shape[0], int(row) + 41)
                c0, c1 = max(0, int(col) - 40), min(normalized.shape[1], int(col) + 41)
                local_means.append(float(normalized[r0:r1, c0:c1].mean()))
            review_ids = {
                int(ids[i]) for i in range(len(ids))
                if original_predictions[i] == 0
                and local_means[i] >= args.review_threshold
            }
            label_slices = ndimage.find_objects(labels)
            for i, label_id in enumerate(ids):
                row, col = centroids[i]
                metrics = hole_metrics(
                    myhc, normalized, labels, int(label_id),
                    args.ring_inner, args.ring_outer,
                    label_slices[int(label_id) - 1],
                )
                normalized_patch = centered_patch(
                    normalized, row, col, args.half_patch_size
                )
                metrics.update({
                    "MF20 Patch Peak (Global Normalized)": float(
                        normalized_patch.max()
                    ),
                    "MF20 Patch Zero Fraction": float(
                        np.mean(normalized_patch == 0)
                    ),
                    "MF20 Patch Saturated Fraction": float(
                        np.mean(normalized_patch == 1)
                    ),
                    "Patch Padding Fraction": patch_padding_fraction(
                        normalized.shape, row, col, args.half_patch_size
                    ),
                })
                nucleus_rows.append({
                    "File": file.name,
                    "Scene": scene,
                    "Label ID": int(label_id),
                    "X": float(col),
                    "Y": float(row),
                    "Model Probability (In)": float(probabilities[i]),
                    "MF20 Local Mean": local_means[i],
                    "Original Class": int(original_predictions[i]),
                    "Final Class": int(final_predictions[i]),
                    "Suggested Review": int(label_id) in review_ids,
                    "Manually Corrected": int(label_id) in corrected_ids,
                    **metrics,
                })

            original_in = int(np.sum(original_predictions == 1))
            n_in = int(np.sum(final_predictions == 1))
            n_total = len(final_predictions)
            fi = 100 * n_in / n_total if n_total else 0.0
            original_fi = 100 * original_in / n_total if n_total else 0.0
            rows.append({
                "File": file.name,
                "Scene": scene,
                "Scene Type": "stitched overview" if scene in overviews else "field",
                "Width": dapi.shape[1],
                "Height": dapi.shape[0],
                "Total Number of Nuclei": n_total,
                "Published MyoFuse Nuclei In": original_in,
                "Published MyoFuse FI (%)": original_fi,
                "Suggested Review Candidates": len(review_ids),
                "Manual Corrections": len(corrected_ids),
                "Nuclei In": n_in,
                "Nuclei Out": int(np.sum(final_predictions == 0)),
                "Fusion Index (%)": fi,
                "Status": "ok" if n_total else "no nuclei detected",
            })
            if args.save_masks:
                imwrite(output / f"{stem}_mask.tif", labels.astype(np.uint32))
            if args.save_overlays:
                save_overlay(output / f"{stem}_prediction.tif", normalized,
                             labels, ids, final_predictions, corrected_ids)
                save_overlay(
                    output / f"{stem}_review_candidates.tif",
                    normalized, labels, ids, original_predictions,
                    review_ids, (0, 210, 255)
                )
            if args.save_patch_qc:
                requested_ids = set(args.qc_label or [])
                if not requested_ids:
                    requested_ids = corrected_ids or review_ids
                save_patch_qc(
                    output / f"{stem}_patch_qc.png",
                    myhc, normalized, labels, ids, original_predictions,
                    probabilities, final_predictions, centroids, requested_ids,
                    args.half_patch_size, args.ring_inner, args.ring_outer,
                )
            print(f"  nuclei={n_total}, in={n_in}, FI={fi:.2f}%, "
                  f"{time.time() - start:.1f}s", flush=True)
            pd.DataFrame(rows).to_excel(output / "Fusion_index.xlsx", index=False)
            pd.DataFrame(nucleus_rows).to_excel(
                output / "Nucleus_predictions.xlsx", index=False
            )

    if not rows:
        raise ValueError("No scenes matched --scene")
    print(f"Results: {output / 'Fusion_index.xlsx'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="CZI file or folder containing CZI files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--corrections", type=Path,
        help=("Optional CSV/XLSX with File, Scene, Label ID, Corrected Class "
              "columns; manual corrections are recorded in the output")
    )
    parser.add_argument("--scene", action="append", help="Only run this scene; repeatable")
    parser.add_argument("--nuclei-channel", type=int, default=None,
                        help="Zero-based index; default: find DAPI by name")
    parser.add_argument("--myotube-channel", type=int, default=None,
                        help="Zero-based index; default: find AF647 by name")
    parser.add_argument("--diameter", type=float, default=24)
    parser.add_argument("--percentile", type=float, default=99.7)
    parser.add_argument(
        "--review-threshold", type=float, default=0.20,
        help=("Flag original-negative nuclei whose normalized local MF20 mean "
              "is at least this value; flags do not change the FI")
    )
    parser.add_argument("--half-patch-size", type=int, default=100)
    parser.add_argument(
        "--save-patch-qc", action="store_true",
        help=("Save a QC contact sheet for --qc-label IDs; without explicit IDs, "
              "use manually corrected nuclei or suggested-review nuclei")
    )
    parser.add_argument(
        "--qc-label", action="append", type=int,
        help="Nucleus Label ID to include in patch QC; repeatable")
    parser.add_argument(
        "--ring-inner", type=int, default=3,
        help="Inner exclusion radius in pixels for the MF20 surrounding ring")
    parser.add_argument(
        "--ring-outer", type=int, default=20,
        help="Outer radius in pixels for the MF20 surrounding ring")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.half_patch_size < 1 or args.threads < 1:
        parser.error("batch size, half patch size, and threads must be positive")
    if not 0 < args.percentile <= 100:
        parser.error("percentile must be in (0, 100]")
    if not 0 <= args.review_threshold <= 1:
        parser.error("review threshold must be in [0, 1]")
    if args.ring_inner < 0 or args.ring_outer <= args.ring_inner:
        parser.error("ring radii must satisfy 0 <= ring-inner < ring-outer")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
