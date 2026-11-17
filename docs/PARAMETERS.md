# Frozen Actin-structure parameters

These values define the automatic method used by the manuscript release. Image
channels are robustly rescaled to 8 bit before the structural thresholds are
applied. Thresholds therefore refer to the normalized 0–255 range, not raw CZI
intensity units.

## MyoFuse stage

| Parameter | Value |
|---|---:|
| Nominal Cellpose nuclear diameter | 24 px |
| MF20 robust upper percentile | 99.7 |
| Nucleus-centred patch | 200 × 200 px |
| Half patch size in code | 100 px |

## Primary MF20/Actin candidate mask

| Parameter | Value |
|---|---:|
| MF20 threshold | 75/255 |
| Actin threshold | 30/255 |
| Actin support radius | 3 px |
| Nuclear support radius | 2 px |
| Minimum nuclear support overlap | 0.15 |
| Minimum nuclei per retained component | 2 |
| Hybrid-only MyoFuse probability guard | 0.024 |

## Actin-structure graph

| Parameter | Value |
|---|---:|
| Maximum nucleus-to-nucleus distance | 60 px |
| Local direction radius | 28 px |
| Minimum Actin coherence | 0.20 |
| Maximum Actin direction difference between nuclei | 35° |
| Maximum Actin-to-connecting-line angle | 30° |
| Actin path threshold | 25/255 |
| MF20 path threshold | 30/255 |
| Minimum Actin-supported path fraction | 0.55 |
| Minimum MF20-supported path fraction | 0.40 |
| Graph path sampling radius | 3 px |

## Adaptive safeguards and review queue

| Parameter | Value |
|---|---:|
| Low original-MyoFuse FI threshold | 0.10 |
| Low-support automatic-In review overlap | 0.25 |
| Close-neighbour review probability | 0.25 |
| Close-neighbour review distance | 35 px |
| Maximum processed mask fraction QC limit | 0.80 |
| Maximum raw mask fraction QC limit | 0.50 |

The browser always activates the Actin-structure flag and does not expose the
later learned classifier or Structure v2 as selectable manuscript methods.

