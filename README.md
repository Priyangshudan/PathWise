# PathWise

PathWise is an Android application project for identifying footpath damage and physical obstructions. The initial computer vision experiment audits and prepares the Indian Footpath Damage Segmentation Dataset for binary semantic segmentation.

## Current status

- Milestone 1: repository setup and dataset audit.
- No model training or Android application code is included.
- See [dataset audit](docs/DATASET_AUDIT.md), [dataset register](docs/DATASET_REGISTER.md), and [architecture](docs/ARCHITECTURE.md).

## Python setup (Windows PowerShell)

```powershell
if (-not (Test-Path .venv)) { py -3.13 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The PyTorch CPU index is used to avoid CUDA packages. Run the audit with:

```powershell
.\.venv\Scripts\python.exe cv\scripts\audit_dataset.py '.\Datasets\Indian Footpath Damage Segmentation Dataset.zip' --output artifacts\dataset_audit\report.json --overlays artifacts\dataset_audit --hash-duplicates
.\.venv\Scripts\python.exe -m unittest discover -s cv\tests -v
```

Dataset source files are kept under `Datasets/` and ignored by Git. Never commit or push the dataset.
