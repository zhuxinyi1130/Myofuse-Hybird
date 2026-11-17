# MyoFuse-Hybrid: Actin-structure manuscript release

This repository contains the frozen **Actin-structure** version of
MyoFuse-Hybrid used for manuscript analysis. It performs automatic fusion-index
(FI) analysis from multichannel skeletal-muscle fluorescence images and
provides a local browser interface for manual quality control.

This release deliberately excludes the later learned classifier, historical
source snapshots, experimental results, and the separate mononucleated-MyHC
tool. The shared analysis engine retains dormant comparison functions for
provenance, but the browser interface exposes and runs only the fixed
Actin-structure manuscript method. Structure v2 and the learned classifier are
not activated by the manuscript interface.

## What the workflow does

1. Reads DAPI/Hoechst, phalloidin/Actin, and MF20/MyHC channels from a CZI file.
2. Segments nuclei with the MyoNuclei Cellpose model released with MyoFuse.
3. Applies the released MyoFuse MF20-based nucleus classifier.
4. Creates candidate MF20/Actin-supported muscle regions.
5. Refines nucleus assignment with an anchored Actin-structure graph. Nearby
   nuclei are linked only when the intervening Actin and MF20 signals are
   sufficiently continuous and the local Actin directions are compatible.
6. Retains multinucleated candidate structures only when at least two valid
   nuclei are associated with the structure.
7. Reports automatic FI and allows manual reassignment, exclusion of
   segmentation artefacts, and addition of missed nuclei.

For each stitched image containing nine imaging fields, the fields are analysed
separately and their nucleus counts are pooled to calculate one image-level FI.
Two stitched images are analysed per well, and their manually corrected
image-level FI values are averaged to obtain one well-level FI:

```text
Image FI (%) = 100 × pooled In nuclei across 9 fields
                   / pooled valid DAPI-positive nuclei across 9 fields

Well FI (%) = (corrected Image 1 FI + corrected Image 2 FI) / 2
```

## Status and scope

- Release: `v1.0.0-manuscript`
- Frozen: 17 November 2026
- Manuscript method: **Actin-structure**
- Tested platform: Apple Silicon Mac, macOS 14.6.1, Python 3.9.6
- Input used in the study: multichannel CZI files
- This is research software and is not a clinical or diagnostic device.
- Automatic classifications require visual quality control. Automatic and
  manually corrected FI values are retained separately.

## Quick start on macOS

1. Download or clone this repository.
2. Double-click `models.zip` and extract it in the repository root.
3. Confirm that the following files are present:
   - `models/MyoNuclei`
   - `models/Svetlana/MyoFuse.pth`
   - `models/Svetlana/Config.json`
4. Double-click `START HERE - MyoFuse Hybrid.command`.
5. On the first launch, approve installation of the Python environment. The
   installation requires an internet connection and may take 10–20 minutes.
6. Drop one CZI file into the browser page or use **Choose file**.
7. Leave **All separate fields (P1–P9)** selected for tiled well acquisitions.
8. Confirm the channel assignment before running a complete dataset.

The extracted folder must be located directly in the repository root. The
correct path is `MyoFuse-Hybrid/models/MyoNuclei`, not
`MyoFuse-Hybrid/models/models/MyoNuclei`.

The local application binds to `127.0.0.1`; image data are processed on the
local computer. New analyses are stored under
`~/Documents/MyoFuse-Hybrid-Analyses/`.

## Expected channels

The application first attempts to identify channels from CZI metadata. Manual
zero-based channel overrides are available in the browser.

| Biological signal | Typical label | Purpose |
|---|---|---|
| Nuclei | DAPI or Hoechst | Nuclear segmentation and FI denominator |
| F-actin | AF488-phalloidin | Local fibre direction and path continuity |
| MyHC | AF647-MF20 | Myogenic signal and MyoFuse classification |

## Manual review

The browser interface preserves the automatic result and records corrections
separately.

- **In**: included in the multinucleated-myotube numerator.
- **Out**: valid nucleus retained in the denominator but not the numerator.
- **Exclude**: invalid or unusable segmentation removed from the corrected
  denominator.
- **Add nucleus**: adds a missed DAPI-positive nucleus for manual classification.

The export contains automatic and corrected FI values, the complete per-nucleus
table, a correction log, and corrected overlays. Display brightness changes
only the viewer and never changes the analysis pixels.

## Command-line use

Run all separate CZI fields with the manuscript settings:

```bash
./run_hybrid_pipeline.sh \
  "/path/to/image.czi" \
  "/path/to/output" \
  ALL \
  0 1 2 \
  0.024 \
  0 0 0 \
  1 0
```

Arguments after the channel indices enable the fixed conservative probability
guard and Actin-structure rescue. The browser interface constructs this command
automatically and is the recommended entry point.

## Repository contents

| Path | Purpose |
|---|---|
| `run_myofuse.py` | Nuclear segmentation and initial MyoFuse classification |
| `run_hybrid_fusion.py` | MF20/Actin candidate regions and Actin-structure graph |
| `run_hybrid_pipeline.sh` | Runs the two analysis stages for one or all fields |
| `manual_review_web.py` | Browser interface and fixed manuscript workflow |
| `manual_review_app.py` | Review state, corrections, and export logic |
| `czi_tiles.py` | CZI scene and channel access |
| `list_czi_scenes.py` | Lists separately acquired CZI fields |
| `models.zip` | Archive containing the required MyoFuse models; extract it in the repository root before first use |
| `CHECKSUMS.sha256` | Integrity checks for bundled model files |
| `docs/METHODS.md` | Paper-ready method description |
| `docs/PARAMETERS.md` | Frozen numerical parameters |
| `docs/USER_GUIDE.md` | Review and export instructions |

## Reproducibility

Use the tagged release `v1.0.0-manuscript` and report the Git commit hash in the
manuscript. The same automatic parameters must be used for every experimental
group. Parameter changes create a different analysis method and should be
validated independently.

No raw microscopy data or manually corrected experimental results are included
in this repository. Representative de-identified input data may be deposited in
a separate research-data archive if permitted by the study's data-sharing plan.

## Relationship to other software

MyoFuse-Hybrid uses the released MyoFuse nucleus-segmentation and nucleus-
classification models. The structural refinement and review application were
implemented independently.

This repository is not an official release from the MyoFuse authors.

## Citation

If this workflow is used, cite the software release, the research article in
which the analysis is reported, and the original MyoFuse publication. Until the
research article is published, cite the repository version and MyoFuse:

> Lair B. et al. MyoFuse is a fully AI-based workflow for automated
> quantification of skeletal muscle cell fusion in vitro. *Scientific Reports*
> 16, 9387 (2026). https://doi.org/10.1038/s41598-026-40047-y

Also cite Cellpose.

## Licences

The independently developed MyoFuse-Hybrid source code is released under the
MIT License in `LICENSE`. The bundled MyoFuse files retain their original MIT
licence and copyright notice in `THIRD_PARTY_MyoFuse_LICENSE`. See
`THIRD_PARTY_NOTICES.md` for attribution details.

