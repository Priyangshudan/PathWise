# Experiment log

Use one section per experiment. Do not record model training until it has actually run.

## Template

### Experiment ID / date
- **Objective:**
- **Dataset version and audit report:**
- **Split used:**
- **Code revision:**
- **Environment:** Python, PyTorch, CPU/GPU
- **Model and configuration:**
- **Preprocessing / augmentations:**
- **Metrics:**
- **Results:**
- **Observations and limitations:**
- **Next action:**

## Initial entry

Milestone 1 was repository setup and dataset audit; no model had been trained at that time.

### U-Net CPU baseline / 2026-11-20
- **Objective:** First binary footpath-damage segmentation baseline.
- **Dataset version and audit report:** Indian Footpath Damage Segmentation Dataset ZIP; `cv/data/experiment_manifest.csv` excludes 18 duplicate-image rows with conflicting masks.
- **Split used:** 402 train, 81 validation; test split held out.
- **Code revision:** Working tree (not committed).
- **Environment:** Python 3.13.7, PyTorch 2.15.0+cpu, CPU.
- **Model and configuration:** Four-level U-Net, base width 16, input 128×128, batch 2, 15 epochs, Adam at 0.001, seed 42.
- **Preprocessing / augmentations:** RGB scaled to [0,1], binary mask threshold 128, nearest-neighbour mask resize, synchronized random horizontal flip on training only.
- **Metrics:** Best validation Dice 0.4324 and IoU 0.2758 at epoch 8 (threshold 0.5).
- **Results:** Final epoch validation Dice 0.3816 and IoU 0.2358. Total training and output time 2447.8 seconds; mean epoch time 161.7 seconds.
- **Observations and limitations:** Small baseline with variable validation metrics; 18 duplicate-image samples were quarantined. Test metrics have not been run.
- **Next action:** Review checkpoint and validation visualizations before any further experiment.
