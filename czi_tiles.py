"""Helpers for CZI files that store fields in a mosaic (M) dimension."""

import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
from aicsimageio import AICSImage


TILE_SCENE_PATTERN = re.compile(r"^P([1-9][0-9]*)$")


def single_scene_mosaic_tiles(
    path: Union[Path, str]
) -> Optional[tuple[str, list[str]]]:
    """Return the native scene and virtual P1..Pn fields for a mosaic-only CZI."""
    image = AICSImage(path, reconstruct_mosaic=False)
    native_scenes = tuple(image.scenes)
    if len(native_scenes) != 1:
        return None
    image.set_scene(native_scenes[0])
    tile_count = int(getattr(image.dims, "M", 1))
    if tile_count <= 1:
        return None
    return native_scenes[0], [f"P{index + 1}" for index in range(tile_count)]


def open_analysis_scene(
    path: Union[Path, str], scene: str
) -> tuple[AICSImage, Optional[int]]:
    """Open a native scene or one virtual Pn mosaic field."""
    native = AICSImage(path, reconstruct_mosaic=True)
    if scene in native.scenes:
        native.set_scene(scene)
        return native, None

    mosaic = single_scene_mosaic_tiles(path)
    match = TILE_SCENE_PATTERN.fullmatch(scene)
    if mosaic is None or match is None:
        available = list(native.scenes)
        if mosaic is not None:
            available = mosaic[1] + available
        raise ValueError(f"Scene {scene!r} not found; available: {tuple(available)}")

    native_scene, tile_scenes = mosaic
    tile_index = int(match.group(1)) - 1
    if tile_index < 0 or tile_index >= len(tile_scenes):
        raise ValueError(f"Scene {scene!r} not found; available: {tuple(tile_scenes)}")
    image = AICSImage(path, reconstruct_mosaic=False)
    image.set_scene(native_scene)
    return image, tile_index


def read_channel(image: AICSImage, channel: int, tile_index: Optional[int]) -> np.ndarray:
    """Read one 2-D channel from a native scene or one mosaic field."""
    selectors = {"T": 0, "Z": 0, "C": channel}
    if tile_index is not None:
        selectors["M"] = tile_index
    return image.get_image_data("YX", **selectors)

