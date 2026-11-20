# PathWise — Persistent Agent Context (`AGENT_CONTEXT.md`)

This document serves as the persistent technical handover and operational source of truth for ChatGPT, Antigravity, and Codex working on the **PathWise** project. Update this document after every completed coding assignment or substantial agent task.

---

## 1. Project overview

- **What PathWise is:** An Android application and computer-vision system for pedestrian accessibility and footpath-condition mapping.
- **Objective:** Identify and map pedestrian footpath damage (cracks, broken/uneven pavement, surface potholes/spalling) and physical obstructions (parked vehicles, poles, bins, barriers) from photographs and video recordings so users can review AI observations, correct predictions, and save geo-tagged infrastructure reports.
- **Intended Android application:** Mobile client allowing users to capture or select images/video, review segmentation/detection overlays, edit observations, store reports locally, and browse mapped accessibility hazards.
- **Computer vision components:**
  1. **Binary semantic segmentation** of footpath surface damage (currently implemented via a compact PyTorch U-Net).
  2. **Object detection** for physical pedestrian obstacles (planned future milestone).
  3. **Image and video processing** pipeline connecting mobile recordings to inference models (planned future milestone).
- **Current development stage:** Milestone 1 (repository setup & dataset audit), Milestone 2 (dataset integrity verification & quarantine manifest), and Milestone 3 (initial CPU U-Net baseline training) are complete. The project is currently in **Model Improvement Phase 1** (baseline diagnosis and controlled U-Net improvements).

---

## 2. Technical architecture

| Component | Status | Details |
|---|---|---|
| **PyTorch (`cv/`)** | **Implemented** | Dataset loader (`cv/data/dataset.py`), compact 4-level U-Net (`cv/models/unet.py`), binary segmentation metrics (`cv/evaluation/metrics.py`), training & evaluation CLI (`cv/training/train_unet.py`), dataset audit script (`cv/scripts/audit_dataset.py`), and unit tests (`cv/tests/`). |
| **Android application (`android/`)** | **Planned** | Placeholder directory (`android/.gitkeep`) only. No Android app code implemented yet. |
| **OpenCV** | **Planned** | Planned for image and video frame processing; not yet installed in `requirements.txt` or implemented. |
| **FastAPI** | **Planned** | Planned local inference API to serve computer-vision predictions; not yet implemented. |
| **Room / SQLite** | **Planned** | Planned local persistence on Android for saving geo-tagged footpath reports; not yet implemented. |
| **MapLibre** | **Planned** | Planned map interface for visualizing footpath damage and obstruction reports; not yet implemented. |

---

## 3. Environment

- **Operating System:** Windows (PowerShell `pwsh` shell).
- **Python version:** Python `3.13.7`.
- **Virtual environment:** Existing `.venv` in repository root (`.\.venv\Scripts\python.exe`).
- **Installed core packages (`requirements.txt`):**
  - `torch==2.15.0+cpu` (installed from `https://download.pytorch.org/whl/cpu`)
  - `numpy>=2.1` (`2.5.5` installed)
  - `Pillow>=11.0` (`13.0.0` installed)
- **Compute hardware:** CPU-only execution, 16 GB system RAM, no NVIDIA GPU / no CUDA packages.
- **Repository structure:**
  ```text
  PathWise/
  ├── .venv/                         # Local Python 3.13.7 virtual environment (ignored by Git)
  ├── AGENT_CONTEXT.md               # Persistent multi-agent context and activity log
  ├── README.md                      # Quickstart and setup instructions
  ├── requirements.txt               # CPU-only Python dependencies
  ├── android/                       # Placeholder for future Android client (.gitkeep)
  ├── Datasets/                      # Raw dataset archive & metadata (ignored by Git)
  │   ├── Indian Footpath Damage Segmentation Dataset.zip
  │   ├── Indian Footpath Damage Segmentation Dataset.md
  │   └── Footpath Damage severity levels_annotations.csv
  ├── artifacts/                     # Audit reports, model checkpoints, plots (ignored by Git)
  │   ├── dataset_audit/             # Audit JSON report and sample overlays
  │   ├── unet_baseline/             # Original 128x128 U-Net baseline checkpoint & logs
  │   └── unet_experiments/          # Controlled U-Net improvement experiments
  ├── cv/                            # Computer vision Python package
  │   ├── configs/                   # Placeholder for configuration files (.gitkeep)
  │   ├── data/
  │   │   ├── dataset.py             # FootpathDamageDataset (ZIP/folder + manifest support)
  │   │   └── experiment_manifest.csv# Derived benchmark manifest excluding 18 duplicate rows
  │   ├── evaluation/
  │   │   └── metrics.py             # Global binary Dice, IoU, precision, recall, F1
  │   ├── inference/                 # Placeholder for inference utilities (.gitkeep)
  │   ├── models/
  │   │   └── unet.py                # 4-level U-Net (ConvBlock + MaxPool + ConvTranspose2d)
  │   ├── scripts/
  │   │   └── audit_dataset.py       # Non-destructive ZIP dataset auditing script
  │   ├── tests/
  │   │   ├── test_dataset.py        # Dataset pairing, rotation, threshold, manifest tests
  │   │   └── test_unet.py           # U-Net forward shape, loss backprop, metric tests
  │   └── training/
  │       └── train_unet.py          # Training loop, plotting, sample sheet, checkpoint eval
  └── docs/                          # Project documentation
      ├── ARCHITECTURE.md
      ├── BASELINE_DIAGNOSIS.md
      ├── DATA_INTEGRITY_REPORT.md
      ├── DATASET_AUDIT.md
      ├── DATASET_REGISTER.md
      ├── EXPERIMENT_LOG.md
      ├── MODEL_EXPERIMENTS.md
      ├── Pathwise_learning_concepts.md
      └── UNET_BASELINE.md
  ```

---

## 4. Dataset information

- **Dataset name:** Indian Footpath Damage Segmentation Dataset.
- **Dataset location:** `Datasets/` (ignored by Git).
- **Original ZIP location:** `Datasets/Indian Footpath Damage Segmentation Dataset.zip` (4,504,525,071 bytes / ~4.20 GB; contains top-level folder `Final Indian Footpath Damage Segmentation Dataset1/` with `Train/`, `Val/`, `Test/`).
- **Adjacent files in `Datasets/`:**
  - `Datasets/Footpath Damage severity levels_annotations.csv` (583 image-level severity labels: 130 low, 262 medium, 191 high).
  - `Datasets/Indian Footpath Damage Segmentation Dataset.md` (dataset description).
- **Image and mask counts:** 583 RGB `.jpg` images paired with 583 grayscale (`L` mode) `.png` masks by numeric filename stem within each split directory.
  - Native image resolutions: 436 at `6000×4000`, 113 at `1500×1000`, 34 at `1800×1200` (all 3:2 landscape aspect ratio).
- **Official split counts:**
  - `Train`: 411 pairs
  - `Val`: 85 pairs
  - `Test`: 87 pairs
  - **Total:** 583 pairs
- **Derived manifest:** `cv/data/experiment_manifest.csv`
  - **Included benchmark counts:** **402 train**, **81 validation**, **82 test** (**565 total**).
  - **Quarantined counts:** **9 train**, **4 validation**, **5 test** (**18 total** across 9 duplicate groups `DG01`–`DG09`).
- **Mask encoding:** Grayscale `L`-mode PNGs with pixel intensities exclusively in two separated clusters: near-black `0–9` (background, ~84.81% of native pixels) and near-white `247–255` (footpath damage, ~15.19% of native pixels), with zero pixels in `10–246`. Thresholding at `>= 128` maps background to `0` and damage to `1`.
- **Orientation correction:** Nine maskspaired with `6000×4000` images are stored at `4000×6000` (transposed width/height). Visual audit verified that rotating these 9 masks 90° counterclockwise (`Image.Transpose.ROTATE_90`) in memory aligns them with their paired images.
- **Duplicate image findings & quarantined samples:** SHA-256 hashing identified 9 byte-identical JPEG pairs (18 files total). Every pair has conflicting ground-truth masks (binary mask IoU ranging from `0.147` to `0.792`), and 7 of the 9 pairs cross official split boundaries (causing train/val/test leakage). One pair (`1092` test / `4008` train) also has conflicting severity labels (`medium` vs `high`). All 18 files are marked `benchmark_status=quarantined` in `cv/data/experiment_manifest.csv` while keeping official split files untouched.
- **Dataset loader location:** `cv/data/dataset.py` (`FootpathDamageDataset`), supporting both direct reading from the original `.zip` archive and reading from extracted directory trees while filtering via `cv/data/experiment_manifest.csv`.

---

## 5. Model and experiment history

### Original U-Net Baseline (`unet_baseline`)
- **Architecture:** Four-level U-Net (`cv/models/unet.py`), `base_channels=16` (encoder/decoder widths `16, 32, 64, 128`, bottleneck `256`), `in_channels=3`, `out_channels=1` (raw logit output).
- **Input resolution:** `128 × 128` (images bilinearly resized, masks nearest-neighbour resized).
- **Batch size:** `2`
- **Epochs:** `15`
- **Optimizer & LR:** `Adam`, learning rate `0.001`
- **Loss function:** Equal-weight BCE-with-logits + soft Dice loss (`0.5 * (BCEWithLogitsLoss + soft Dice loss)`, `smooth=1.0`).
- **Random seed:** `42`
- **Augmentation:** Random horizontal flip (`p=0.5`) on training split only.
- **Best checkpoint:** Epoch `8` (`artifacts/unet_baseline/best_model.pth`)
- **Best validation metrics (threshold `0.5`):**
  - **Validation Dice (F1):** `0.4324` (`0.432351`)
  - **Validation IoU:** `0.2758` (`0.275796`)
  - **Validation Precision:** `0.3919` (`0.391879`)
  - **Validation Recall:** `0.4821` (`0.482144`)
  - **Validation Loss:** `0.5729` (`0.572905`)
- **Final epoch (Epoch 15) validation metrics:** Dice `0.3816`, IoU `0.2358`, validation loss `0.5463`.
- **Training runtime:** `2447.81` seconds (~`40` minutes `48` seconds; average `161.68` seconds/epoch) reading directly from the 4.2 GB ZIP archive on CPU.
- **Artifacts:**
  - Checkpoint: `artifacts/unet_baseline/best_model.pth`
  - History: `artifacts/unet_baseline/history.csv`, `artifacts/unet_baseline/history.json`
  - Summary: `artifacts/unet_baseline/training_summary.json`
  - Plots: `artifacts/unet_baseline/training_curves.png`
  - Predictions visual sheet: `artifacts/unet_baseline/samples/validation_predictions.png`

### Phase 1 Controlled U-Net Improvements (`artifacts/unet_experiments/`)
All Phase 1 experiments used `256 × 256` input resolution (bilinear image resize, nearest-neighbour mask resize), `base_channels=16`, `seed=42`, `cv/data/experiment_manifest.csv` (`402` train, `81` val), and on-disk resized PNG caching (`artifacts/dataset_cache/`):

| Experiment ID | Batch / Epochs (Best) | Augmentations & Scheduler | Loss Function | Val Dice (`t=0.50`) | Val IoU (`t=0.50`) | Val Prec / Rec (`t=0.50`) | Selected `t*` | Val Dice (`t*`) | Val IoU (`t*`) | Runtime (Avg/Ep) |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| **`exp1_res256`** | `batch=2`, `15` ep (Best: Ep `4`) | H-flip (`0.5`), Fixed `lr=0.001` | `0.50 BCE + 0.50 sample Dice` | `0.4258` | `0.2705` | `0.4329` / `0.4189` | `0.30` | `0.4452` | `0.2864` | `1428.5s` (`94.6s/ep`) |
| **`exp2_train_improvements`** | `batch=4`, `18` ep (Best: Ep `17`) | H-flip (`0.5`), Rot (`±10°`), `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.50 BCE + 0.50 sample Dice` | `0.4775` | `0.3137` | **`0.4578`** / `0.4991` | `0.30` | **`0.4888`** | **`0.3235`** | `1577.9s` (`87.1s/ep`) |
| **`exp3a_dice_focused`** | `batch=4`, `18` ep (Best: Ep `15`) | H-flip (`0.5`), Rot (`±10°`), `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.20 BCE + 0.80 batch Dice` | `0.4737` | `0.3103` | `0.4337` / `0.5217` | `0.30` | `0.4796` | `0.3155` | `1456.7s` (`80.4s/ep`) |
| **`exp3b_modest_weight`** | `batch=4`, `18` ep (Best: Ep `14`) | H-flip (`0.5`), Rot (`±10°`), `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.35 BCE + 0.65 batch Dice` | **`0.4845`** | **`0.3197`** | `0.4072` / **`0.5980`** | `0.40` | `0.4866` | `0.3215` | `1324.3s` (`73.1s/ep`) |

---

## 6. Current development status

- **Completed milestones:**
  1. Milestone 1: Repository structure, environment setup, and non-destructive dataset audit (`docs/DATASET_AUDIT.md`, `docs/DATASET_REGISTER.md`).
  2. Milestone 2: Duplicate-image leak verification, mask/orientation verification, and derived benchmark manifest (`cv/data/experiment_manifest.csv`, `docs/DATA_INTEGRITY_REPORT.md`).
  3. Milestone 3: Compact CPU U-Net implementation, unit tests, and 15-epoch baseline training (`artifacts/unet_baseline/`, `docs/UNET_BASELINE.md`, `docs/EXPERIMENT_LOG.md`).
  4. Milestone 4 (Model Improvement Phase 1): Persistent agent context (`AGENT_CONTEXT.md`), baseline diagnosis (`docs/BASELINE_DIAGNOSIS.md`), on-disk resized dataset caching (`artifacts/dataset_cache/`), live Markdown experiment progress tracker (`artifacts/unet_experiments/EXPERIMENT_PROGRESS.md`), and Controlled Experiments 1–4 (`docs/MODEL_EXPERIMENTS.md`).
- **Active milestone:** Model Improvement Phase 1 complete — awaiting user review and approval before next phase (e.g., held-out test split evaluation or inference/API integration).
- **Pending tasks:**
  - Upon user approval, evaluate the selected best checkpoint (`exp2_train_improvements` or `exp3b_modest_weight`) and the original baseline on the held-out test split (`82` included test samples).
  - Subsequent project milestones: inference script/API (`cv/inference/`, FastAPI), obstacle object detection, video frame processing, and Android application integration.
- **Known issues:**
  - The compact scratch U-Net (`base_channels=16`) still has a modest ceiling (`~0.489` validation Dice) due to small training sample size (`402` images), lack of pretrained encoder features, and mixed polygon-vs-crack annotation styles in the source dataset.
  - Hairline cracks (`< 10px` at `6000 × 4000`) remain ~1px wide at `256 × 256`.
- **Current best model:**
  - **Best at fixed `0.50` threshold:** `artifacts/unet_experiments/exp3b_modest_weight/best_model.pth` (Epoch 14: Validation Dice **`0.4845`**, IoU **`0.3197`** at `t=0.50`; Dice **`0.4866`**, IoU **`0.3215`** at `t*=0.40`).
  - **Best overall with validation-selected threshold (`t*=0.30`) & highest precision:** `artifacts/unet_experiments/exp2_train_improvements/best_model.pth` (Epoch 17: Validation Dice **`0.4888`**, IoU **`0.3235`** at `t*=0.30`; Dice **`0.4775`**, IoU **`0.3137`**, Precision **`0.4578`** at `t=0.50`).
- **Latest validation metrics:**
  - `exp2_train_improvements` (Epoch 17): Fixed `t=0.50` -> Dice `0.4775`, IoU `0.3137`, Precision `0.4578`, Recall `0.4991`; Tuned `t*=0.30` -> Dice **`0.4888`**, IoU **`0.3235`**, Precision `0.4184`, Recall `0.5877`.
  - `exp3b_modest_weight` (Epoch 14): Fixed `t=0.50` -> Dice **`0.4845`**, IoU **`0.3197`**, Precision `0.4072`, Recall `0.5980`; Tuned `t*=0.40` -> Dice `0.4866`, IoU `0.3215`, Precision `0.3943`, Recall `0.6352`.
- **Latest test metrics:** Not yet evaluated (`test_metrics.json` not generated; test split kept strictly held out during Phase 1 tuning).

---

## 7. Agent activity log

### 2026-11-20 — Antigravity — Model Improvement Phase 1 (Parts C, D, E, F: Caching, Controlled Experiments 1–4 & Documentation)
- **Date:** 2026-11-20
- **Agent name:** Antigravity
- **Task/prompt title:** PathWise — Model Improvement Phase 1 + Persistent Agent Context (Parts C–F)
- **Objective:** Implement non-destructive local dataset caching for CPU efficiency, add live experiment progress tracking and auto-resume support, execute Controlled Experiments 1–4 (`256×256` resolution, training/augmentation/scheduler improvements, loss function comparison, validation threshold selection), run the unit test suite, and document all findings in `docs/MODEL_EXPERIMENTS.md` and `AGENT_CONTEXT.md`.
- **Files created:**
  - `docs/MODEL_EXPERIMENTS.md`
  - `artifacts/unet_experiments/EXPERIMENT_PROGRESS.md` (git-ignored live tracker)
  - Experiment outputs under `artifacts/unet_experiments/exp1_res256/`, `exp2_train_improvements/`, `exp3a_dice_focused/`, and `exp3b_modest_weight/` (git-ignored)
- **Files modified:**
  - `cv/data/dataset.py` (added optional `cache_dir`, `vertical_flip`, and `max_rotation_degrees` with bilinear image and nearest-neighbour mask resampling)
  - `cv/training/train_unet.py` (added configurable `bce_weight`, `dice_weight`, `batch_dice`, `ReduceLROnPlateau`, early stopping, validation threshold sweep `threshold_sweep.json`, intra-epoch `progress.json` + live Markdown progress updater `update_progress_markdown()`, and per-epoch `last_checkpoint.pth` `--resume` support)
  - `cv/tests/test_dataset.py` (added `test_disk_cache_and_rotation_preserve_binary_masks`)
  - `cv/tests/test_unet.py` (added `test_weighted_losses_and_threshold_sweep`)
  - `docs/BASELINE_DIAGNOSIS.md` (added Section 10 post-experiment verification summary)
  - `AGENT_CONTEXT.md` (updated experiment history, current status, and agent activity log)
- **Important implementation decisions:**
  - Cached orientation-corrected resized PNG pairs under `artifacts/dataset_cache/<HxW>_t<threshold>/<Split>/` (ignored by Git) so the original 4.2 GB ZIP is never modified and `256 × 256` epochs run in `73–87s` instead of `~220s`.
  - Used horizontal flip (`p=0.5`) and small-angle rotation (`±10°`) for training augmentation; left vertical flip disabled by default because upside-down oblique footpath views are unnatural.
  - Added batch-level soft Dice option (`--batch-dice`) in Experiment 3 so images with tiny (`0.1%`) ground-truth cracks do not produce unstable single-sample Dice gradients.
  - Added automatic per-epoch `last_checkpoint.pth` and live Markdown progress table updates so CPU training runs can be monitored in real time and resumed seamlessly after interruptions.
- **Commands executed:**
  - `.\.venv\Scripts\python.exe -m unittest discover -s cv\tests -v`
  - `.\.venv\Scripts\python.exe cv\training\train_unet.py --experiment-name exp1_res256 --input-size 256 --batch-size 2 --epochs 15 --learning-rate 0.001 --base-channels 16 --seed 42 --cache-dir artifacts\dataset_cache --output artifacts\unet_experiments\exp1_res256`
  - `.\.venv\Scripts\python.exe cv\training\train_unet.py --experiment-name exp2_train_improvements --input-size 256 --batch-size 4 --epochs 18 --learning-rate 0.001 --base-channels 16 --rotation-degrees 10.0 --lr-scheduler --lr-patience 2 --lr-factor 0.5 --early-stopping-patience 5 --seed 42 --cache-dir artifacts\dataset_cache --output artifacts\unet_experiments\exp2_train_improvements`
  - `.\.venv\Scripts\python.exe cv\training\train_unet.py --experiment-name exp3a_dice_focused --input-size 256 --batch-size 4 --epochs 18 --learning-rate 0.001 --base-channels 16 --bce-weight 0.2 --dice-weight 0.8 --batch-dice --rotation-degrees 10.0 --lr-scheduler --lr-patience 2 --lr-factor 0.5 --early-stopping-patience 5 --seed 42 --cache-dir artifacts\dataset_cache --output artifacts\unet_experiments\exp3a_dice_focused --resume`
  - `.\.venv\Scripts\python.exe cv\training\train_unet.py --experiment-name exp3b_modest_weight --input-size 256 --batch-size 4 --epochs 18 --learning-rate 0.001 --base-channels 16 --bce-weight 0.35 --dice-weight 0.65 --batch-dice --rotation-degrees 10.0 --lr-scheduler --lr-patience 2 --lr-factor 0.5 --early-stopping-patience 5 --seed 42 --cache-dir artifacts\dataset_cache --output artifacts\unet_experiments\exp3b_modest_weight --resume`
- **Tests executed and results:** 17 unit tests in `cv/tests` (`test_dataset.py` and `test_unet.py`); all 17 passed (`OK`).
- **Experiments conducted:** `exp1_res256`, `exp2_train_improvements`, `exp3a_dice_focused`, `exp3b_modest_weight`, and validation threshold sweeps (`0.30–0.70`) on all checkpoints.
- **Metrics obtained:**
  - `unet_baseline` (`128×128`): `t=0.50` Dice `0.4324`, IoU `0.2758`; `t*=0.40` Dice `0.4345`, IoU `0.2775` (`2447.8s`).
  - `exp1_res256` (`256×256`): `t=0.50` Dice `0.4258`, IoU `0.2705`, Precision `0.4329`; `t*=0.30` Dice `0.4452`, IoU `0.2864` (`1428.5s`).
  - `exp2_train_improvements` (`256×256`, `0.50 BCE + 0.50 Dice`): `t=0.50` Dice **`0.4775`**, IoU **`0.3137`**, Precision **`0.4578`**, Recall `0.4991`; `t*=0.30` Dice **`0.4888`**, IoU **`0.3235`** (`1577.9s`).
  - `exp3a_dice_focused` (`256×256`, `0.20 BCE + 0.80 batch Dice`): `t=0.50` Dice `0.4737`, IoU `0.3103`; `t*=0.30` Dice `0.4796`, IoU `0.3155` (`1456.7s`).
  - `exp3b_modest_weight` (`256×256`, `0.35 BCE + 0.65 batch Dice`): `t=0.50` Dice **`0.4845`**, IoU **`0.3197`**, Recall **`0.5980`**; `t*=0.40` Dice `0.4866`, IoU `0.3215` (`1324.3s`).
- **Known issues:** Scratch U-Net capacity and mixed ground-truth annotation granularity cap pixel-level IoU around `0.32` at `256 × 256`.
- **Git status:** Modified tracked files: `cv/data/dataset.py`, `cv/training/train_unet.py`, `cv/tests/test_dataset.py`, `cv/tests/test_unet.py`; untracked documentation files: `AGENT_CONTEXT.md`, `docs/BASELINE_DIAGNOSIS.md`, `docs/MODEL_EXPERIMENTS.md`. No commits or pushes performed.
- **Next recommended task:** Review Phase 1 results and, once approved, run a single held-out test evaluation comparing the original baseline checkpoint (`artifacts/unet_baseline/best_model.pth`) against the best Phase 1 checkpoint (`exp2_train_improvements` / `exp3b_modest_weight`).

### 2026-11-20 — Antigravity — Model Improvement Phase 1 (Part A & Part B: Context & Baseline Diagnosis)
- **Date:** 2026-11-20
- **Agent name:** Antigravity
- **Task/prompt title:** PathWise — Model Improvement Phase 1 + Persistent Agent Context (Parts A & B)
- **Objective:** Inspect existing repository, run unit tests, establish `AGENT_CONTEXT.md`, and perform a quantitative and visual diagnosis of the U-Net baseline in `docs/BASELINE_DIAGNOSIS.md` before making code modifications.
- **Files created:**
  - `AGENT_CONTEXT.md`
  - `docs/BASELINE_DIAGNOSIS.md`
- **Files modified:** None (existing code and baseline artifacts untouched).
- **Important implementation decisions:**
  - Verified that no held-out test evaluation had been run (`test_metrics.json` absent); kept test set untouched during development.
  - Profiled data loading vs neural network compute on CPU: found that decoding and resizing 483 high-resolution (`6000×4000`) pairs from the ZIP takes ~140s per epoch, whereas 128×128 U-Net forward+backward takes ~20s per epoch and 256×256 takes ~76s per epoch.
- **Commands executed:**
  - `.\.venv\Scripts\python.exe -m unittest discover -s cv\tests -v`
  - Diagnostic profiling script in conversation scratch directory to measure ZIP I/O speed, validation class balance, per-sample Dice distribution, and validation threshold sweep.
- **Tests executed and results:** 15 unit tests in `cv/tests` executed; all 15 passed in `0.794s`.
- **Experiments conducted:** Baseline diagnostic analysis and validation threshold sweep on existing checkpoint `artifacts/unet_baseline/best_model.pth`.
- **Metrics obtained:**
  - Verified baseline validation metrics at threshold `0.50`: Dice `0.4324`, IoU `0.2758`, Precision `0.3919`, Recall `0.4821`.
  - Validation pixel proportions at `128×128`: ground-truth damage is `15.62%` of pixels (per-image median `9.25%`, range `0.04%–77.69%`), while baseline predictions at `0.50` cover `19.22%` of pixels (per-image median `20.72%`, range `0.00%–33.05%`).
- **Known issues:** ZIP decompression + 6000×4000 decoding dominates epoch time; 128×128 predictions are blurry central blobs; validation metrics oscillate under constant LR `0.001`.
- **Git status:** Working tree clean on `main` (`8dea604 Baseline model`) prior to adding `AGENT_CONTEXT.md` and `docs/BASELINE_DIAGNOSIS.md`.
- **Next recommended task:** Proceed with Parts C, D, E, and F — implement non-destructive local dataset caching for computational efficiency and execute Controlled Experiments 1–4 under `artifacts/unet_experiments/`.

---

## 8. Important development rules

1. **Preserve working code:** Do not rewrite or break existing tested modules; keep interfaces backwards-compatible.
2. **Never overwrite the original baseline checkpoint:** `artifacts/unet_baseline/best_model.pth` and its companion baseline files must remain intact as the permanent comparison reference.
3. **Never modify original dataset files or official splits:** `Datasets/Indian Footpath Damage Segmentation Dataset.zip` and official split definitions must never be altered.
4. **Use the derived experiment manifest:** Always pass `cv/data/experiment_manifest.csv` so split assignments are verified and quarantined duplicates are excluded.
5. **Keep quarantined samples excluded:** All 18 duplicate-image rows (`DG01`–`DG09`) must remain excluded from training, validation, and primary benchmarking.
6. **Do not use the test set for tuning:** Never select hyperparameters, loss weights, architectures, or probability thresholds using test split performance.
7. **Avoid unnecessary dependencies:** Stick to the lightweight CPU stack (`torch`, `numpy`, `Pillow`, standard library). Do not add CUDA, heavy frameworks, or paid services.
8. **Prefer reproducible experiments:** Fix random seeds (`seed_everything(42)`), log exact configurations, and save structured JSON/CSV summaries for every run.
9. **Do not commit or push without approval:** Leave Git commits and pushes to explicit user instruction.
10. **Keep changes focused and incremental:** Test one controlled factor or modest combination at a time and verify with unit tests before and after changes.
