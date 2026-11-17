# Suggested manuscript Methods

## Semi-automated fusion index analysis

Fusion index (FI) was quantified using MyoFuse-Hybrid
(`v1.0.0-manuscript`), a custom Python-based image-analysis workflow combining
the previously published MyoFuse models with independently implemented MyHC-
and actin-based structural classification. Multichannel CZI images containing
DAPI, AF488-phalloidin, and AF647-MF20 signals were used to identify nuclei,
F-actin structures, and MyHC-positive myogenic structures, respectively. Each
microscopic field was analysed independently.

Nuclei were segmented using the MyoNuclei Cellpose model provided with
MyoFuse, with a nominal nuclear diameter of 24 pixels. For each detected
nucleus, the MyoFuse classifier analysed a 200 × 200-pixel MF20 image patch
together with the corresponding nuclear mask to predict whether the nucleus was
located inside or outside a myotube. Nucleus assignment was subsequently
refined using MF20 and actin structural information. In Actin-structure mode,
each nucleus was represented as a node in a local graph, and nearby nuclei were
connected only when the intervening actin and MF20 signals showed sufficient
spatial continuity and were compatible with the local actin-fibre orientation.
Candidate graph components were required to contain an initial MyoFuse-Hybrid
In anchor. Actin-positive connections lacking sufficient MF20 support were
rejected. Multinucleated candidate structures were required to contain at least
two valid nuclei.

FI was calculated as the number of nuclei assigned to retained multinucleated
candidate structures divided by the total number of valid DAPI-positive nuclei,
expressed as a percentage. Nine fields within each stitched image were processed
independently, and their nuclear counts were pooled to calculate one image-level
FI. Two stitched images were analysed per well, and their corrected image-level
FI values were averaged to obtain one FI value per well.

All automated results were visually inspected using the software's interactive
review interface. Incorrect classifications were manually reassigned as In or
Out, segmentation artefacts were excluded, and missed nuclei were added where
necessary. FI was automatically recalculated following manual correction.
Automated and manually corrected FI values were retained separately, with the
corrected FI values used for subsequent statistical analysis. The same
automatic analysis parameters were applied across all experimental groups.

## Reporting checklist

Before submission, replace or add the following information in the manuscript:

- GitHub release URL and Git commit hash.
- Permanent Zenodo DOI, if available.
- MyoFuse and Cellpose citations.
- Whether the analyst was blinded to treatment condition, but only if this was
  actually done.
- The number of biological replicates and the definition of one statistical
  replicate.
- Confirmation that the average of the two stitched-image FI values, rather
  than either image alone, was used as the well-level value.
- A statement identifying corrected FI, rather than automatic FI, as the value
  used for statistical analysis.

