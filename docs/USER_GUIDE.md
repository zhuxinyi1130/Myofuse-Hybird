# User guide

## Starting the application

On macOS, double-click `START HERE - MyoFuse Hybrid.command`. Keep the Terminal
window open while the browser interface is running. Closing the Terminal window
stops the local server.

## Running a new CZI analysis

1. Drag one CZI file into the upload area or select it with **Choose file**.
2. Confirm that the fixed method is labelled **Actin-structure — Manuscript
   method**.
3. For a nine-field acquisition, select **All separate fields (P1–P9)**.
4. Leave channel overrides blank when the CZI metadata clearly identifies
   DAPI, AF488 Actin, and AF647 MF20. Otherwise enter zero-based indices.
5. Select **Run analysis** and wait for all fields to finish.

## Reviewing classifications

- Green outlines: automatic or manually confirmed In nuclei.
- Red outlines: Out nuclei.
- Yellow outlines: review candidates excluded from automatic FI until resolved.
- Cyan outline: retained candidate muscle-region boundary.

Click anywhere inside a nuclear mask to select it. Drag a rectangle to select
multiple nuclei. Use **Set In**, **Set Out**, or **Exclude** for the selected
nuclei. Use **Add nucleus mode** to place a missed nucleus at a DAPI-positive
location. Undo and Redo are available during the review.

Brightness and channel-display controls affect only visualization. They do not
modify the pixels used by the analysis.

## Exporting

Export after every field has been checked. For a multi-field CZI, the export
contains one combined workbook and a combined visual summary. The nine field
counts are pooled to calculate one corrected FI for that stitched image.

Analyse the second stitched image from the same well in the same way. The
software does not automatically combine two separate CZI files. During
downstream data aggregation, calculate the well-level value as:

```text
Well FI (%) = (corrected Image 1 FI + corrected Image 2 FI) / 2
```

Retain both:

- **Automatic FI**, for evaluation of the automatic method; and
- **Corrected FI**, for the final manually quality-controlled result.

Do not replace an automatic workbook manually in Excel. Corrections should be
made in the review interface so that the audit trail is preserved.

