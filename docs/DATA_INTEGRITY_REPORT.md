# Dataset integrity check

## Duplicate images and quarantine

Targeted verification of the nine known byte-identical image pairs confirmed that every pair has different decoded masks. Seven pairs cross official splits. The manifest quarantines each whole duplicate group because its masks conflict; official split fields remain unchanged.

| Group | IDs and official splits | Severity | Binary mask IoU | Result |
|---|---|---|---:|---|
| DG01 | 1088 test / 4002 test | medium / medium | 0.545 | Quarantined |
| DG02 | 1089 test / 4003 train | medium / medium | 0.754 | Quarantined |
| DG03 | 1091 test / 4005 train | medium / medium | 0.147 | Quarantined |
| DG04 | 1092 test / 4008 train | medium / high | 0.598 | Quarantined; severity conflict |
| DG05 | 4020 train / 1102 validation | medium / medium | 0.792 | Quarantined |
| DG06 | 1086 train / 4000 train | high / high | 0.466 | Quarantined |
| DG07 | 1094 train / 4010 validation | low / low | 0.254 | Quarantined |
| DG08 | 1097 train / 4014 validation | medium / medium | 0.256 | Quarantined |
| DG09 | 4009 train / 1093 validation | medium / medium | 0.648 | Quarantined |

The 18 affected rows are recorded in `cv/data/experiment_manifest.csv` with their original split, group ID, and exclusion reason. Seven training copies that cross into validation/test are excluded. The duplicate test pair is quarantined, so the same test image is not counted twice. No samples were moved between official splits.

| Official split | Original | Quarantined | Included in primary benchmark |
|---|---:|---:|---:|
| Train | 411 | 9 | 402 |
| Validation | 85 | 4 | 81 |
| Test | 87 | 5 | 82 |
| **Total** | **583** | **18** | **565** |

## Mask and orientation checks

Reviewed overlays include binary-only masks and `KGD02064`, which contains intermediate grayscale edge values. Across the archive, values remain in `0–9` or `247–255`, with no values between those groups. Threshold 128 therefore separates the observed background and damage clusters cleanly; the existing loader threshold is retained.

Reviewed all nine reversed-dimension samples in `artifacts/dataset_audit/orientation_review.jpg`. Rotating each mask 90° counterclockwise aligned it with its image. The existing loader correction is retained.

## Loader and remaining issues

`FootpathDamageDataset` now accepts the derived manifest for either a dataset directory or the original ZIP. It selects only `included` rows in the requested official split. Image tensors are float32 CHW; mask tensors are int64 HW containing only 0/1. Pairing and post-orientation dimensions are checked, and resizing uses nearest-neighbour interpolation for masks.

No remaining issue blocks a baseline experiment using the 565 included samples. The 18 quarantined annotations should stay out of the primary benchmark until reviewed.
