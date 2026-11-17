#!/usr/bin/env python3
"""List independent analysis fields, including CZI mosaic M-dimension tiles."""

import argparse
import json
import statistics
from pathlib import Path

from aicsimageio import AICSImage

from czi_tiles import single_scene_mosaic_tiles


def inspect(path: Path) -> dict:
    image = AICSImage(path, reconstruct_mosaic=True)
    native_scenes = list(image.scenes)
    mosaic = single_scene_mosaic_tiles(path)
    if mosaic is not None:
        native_scene, fields = mosaic
        overviews = [native_scene]
        scenes = fields + overviews
    else:
        areas = []
        for scene in native_scenes:
            image.set_scene(scene)
            areas.append(int(image.dims.Y) * int(image.dims.X))
        median = statistics.median(areas) if areas else 0
        overviews = [
            scene for scene, area in zip(native_scenes, areas)
            if median and area > 4 * median
        ]
        scenes = native_scenes
        fields = [scene for scene in scenes if scene not in overviews]
    image.set_scene(native_scenes[0])
    return {
        "channels": list(image.channel_names),
        "scenes": scenes,
        "field_scenes": fields,
        "overview_scenes": overviews,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    metadata = inspect(args.input)
    if args.json:
        print("METADATA_JSON=" + json.dumps(metadata))
    else:
        print("\n".join(metadata["field_scenes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

