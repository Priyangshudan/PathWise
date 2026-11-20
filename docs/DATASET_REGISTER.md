# Dataset register

| Field | Record |
|---|---|
| Name | Indian Footpath Damage Segmentation Dataset |
| Source URL / DOI | Not present in supplied local metadata; unresolved |
| Purpose | Footpath damage segmentation and severity research, according to the supplied description |
| Licence | Not specified in the ZIP, Markdown, or CSV; unresolved. Redistribution terms must be confirmed with the source. |
| Archive | `Datasets/Indian Footpath Damage Segmentation Dataset.zip` (preserved, not extracted) |
| Images / masks | 583 JPEG images and 583 grayscale PNG masks; 583 filename-stem pairs |
| Official splits | Train 411, Val 85, Test 87 |
| Annotation format | Grayscale pixel-wise PNG masks; local Markdown says polygons were created in LabelMe, but LabelMe JSON annotations are not included in the archive |
| Mask encoding | Observed values `0–9` and `247–255`; loader maps below 128 to background and 128+ to damage based on visual inspection and separated value clusters |
| Severity metadata | Adjacent CSV contains 583 image-level labels: low 130, medium 262, high 191; all image stems match archive JPEGs |
| Attribution requirements | Not supplied; unresolved with source and licence |
| Known limitations | Nine masks are dimension-reversed relative to paired images and need a visually verified 90° counterclockwise in-memory rotation. Nine pairs of identical JPEGs have differing masks; seven pairs cross official splits. One duplicate pair has conflicting severity labels. The local description does not give the mask convention, numeric crack-width severity thresholds, collection protocol, source, or licence. |

Numerical findings are from the local archive and adjacent CSV, recorded in [dataset audit](DATASET_AUDIT.md) and `artifacts/dataset_audit/report.json`. Descriptive claims in the supplied Markdown (such as expert validation and model accuracy) are not independently verified.
