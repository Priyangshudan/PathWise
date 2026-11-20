# Dataset audit

## Scope and method

The ZIP `Datasets/Indian Footpath Damage Segmentation Dataset.zip` was inspected directly; it was not extracted or modified. All image and mask entries were decoded one at a time. Pairing uses a common directory and filename stem. The full machine-readable report is `artifacts/dataset_audit/report.json`; sample overlays and an orientation review are in the same directory. Generated outputs are ignored by Git.

## Verified structure and counts

- One top-level directory: `Final Indian Footpath Damage Segmentation Dataset1/`.
- Official split directories: `Train/`, `Val/`, `Test/`.
- 583 JPEG images and 583 grayscale PNG masks; all 583 pairs match by same split directory and numeric filename stem.
- Split counts: train 411, validation 85, test 87. These agree with the supplied Markdown description.
- Image dimensions: 113 at 1500×1000, 34 at 1800×1200, and 436 at 6000×4000.
- Mask dimensions: 113 at 1500×1000, 34 at 1800×1200, 427 at 6000×4000, and 9 at 4000×6000.
- Nine masks have width and height reversed relative to their paired 6000×4000 images. A visual review of all nine showed alignment after a 90° counterclockwise rotation of the mask. Originals remain unchanged; the loader applies this transform in memory only for that exact reversed-dimension case.
- No missing masks, unmatched masks, ambiguous filename pairs, or corrupt image files were found.
- The archive contains only `.jpg` and `.png` file extensions; no CSV, LabelMe JSON, or other metadata files are inside the ZIP.

## Mask encoding and binary conversion

All 583 masks use grayscale `L` mode. The unique pixel values across the archive are `0–9` and `247–255`; no values occur in the wide middle interval `10–246`. This produces two clearly separated near-black and near-white value clusters. Sample overlays show the high-valued cluster marking the damage regions. The dataset loader maps values below 128 to background label `0` and values at or above 128 to damage label `1`. This threshold is an explicit conversion inference from the observed value gap and visual review, not a documented dataset convention.

## Adjacent severity metadata

`Datasets/Footpath Damage severity levels_annotations.csv` is beside the ZIP, not inside it. It contains 583 rows with columns `Image_name` and `Footpath Damage Severity levels`. All 583 image stems match JPEGs in the archive. The class counts are low 130, medium 262, and high 191. These are image-level severity labels and do not encode the pixel-wise mask. The supplied Markdown says a `footpath_damage_metadata.csv` is included, but that filename is absent from the archive and differs from the adjacent CSV's name. The supplied Markdown claims expert crack-width criteria but gives no numeric thresholds or cited source.

## Duplicate and visual checks

Files sharing the same byte size were SHA-256 hashed. This found nine pairs of byte-identical JPEG images (18 files total). Seven pairs cross the official split boundary, so the current split assignments contain exact-image leakage. The audit did not change the official splits. For all nine duplicate-image pairs, the corresponding masks differ; their binary mask IoU ranges from 0.147 to 0.792. One pair (1092 / 4008) also has conflicting severity labels (`medium` / `high`). These findings can affect evaluation and require source-level clarification before interpreting model scores.

The full hashes and paths are in `duplicate_content` in the JSON report. Four red-tinted sample overlays were reviewed, one additional overlay includes intermediate grayscale values, and all nine reversed-dimension pairs were reviewed in `orientation_review.jpg`. The masks align after the stated in-memory rotation.

## Assumptions and unresolved questions

- The `>=128` binary conversion is supported by the large gap between the two observed grayscale clusters and visual alignment. The archive does not explicitly state the threshold or binary label names.
- Nine annotations appear transposed; the required counterclockwise rotation is visually consistent across all nine examples but is not described by the dataset publisher.
- Exact duplicate images occur across splits with different masks (and one severity-label conflict); it is unclear whether this reflects annotation inconsistency, distinct versions of an annotation, or another dataset issue.
- The supplied local metadata does not provide a verifiable source URL, DOI, licence, or redistribution terms. See [dataset register](DATASET_REGISTER.md).
