# U-Net Baseline Diagnosis (`docs/BASELINE_DIAGNOSIS.md`)

## 1. Overview of Inspected Artifacts

Before implementing any model or training changes, the original U-Net baseline (`artifacts/unet_baseline/`) and dataset audit findings (`artifacts/dataset_audit/report.json`, `cv/data/experiment_manifest.csv`) were inspected:

- **Checkpoint:** `artifacts/unet_baseline/best_model.pth` (saved at Epoch 8 of 15)
- **Training history:** `artifacts/unet_baseline/history.csv` and `history.json`
- **Summary:** `artifacts/unet_baseline/training_summary.json`
- **Visual plots:** `artifacts/unet_baseline/training_curves.png` and `artifacts/unet_baseline/samples/validation_predictions.png`
- **Held-out test status:** Verified that `artifacts/unet_baseline/test_metrics.json` **does not exist**. No test-set evaluation has been performed yet. In accordance with project rules, the test split (`82` included samples) is kept strictly separate from all model development and tuning.

---

## 2. Training and Validation Loss Curves

From `artifacts/unet_baseline/history.csv` and `training_curves.png`:

| Epoch | Train Loss | Validation Loss | Validation Dice | Validation IoU | Epoch Time (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6424 | 0.6483 | 0.3617 | 0.2208 | 149.6 |
| 2 | 0.5955 | 0.6004 | 0.1696 | 0.0926 | 158.3 |
| 3 | 0.5772 | 0.6451 | 0.1453 | 0.0784 | 148.4 |
| 4 | 0.5688 | 0.6456 | 0.4028 | 0.2522 | 167.6 |
| 5 | 0.5645 | 0.5948 | 0.3132 | 0.1857 | 168.9 |
| 6 | 0.5652 | 0.5570 | 0.3610 | 0.2202 | 169.1 |
| 7 | 0.5552 | 0.5598 | 0.3699 | 0.2269 | 168.7 |
| **8 (Best)** | **0.5538** | **0.5729** | **0.4324** | **0.2758** | **166.9** |
| 9 | 0.5515 | 0.5717 | 0.3147 | 0.1867 | 168.1 |
| 10 | 0.5445 | 0.5810 | 0.3512 | 0.2130 | 166.9 |
| 11 | 0.5446 | 0.6105 | 0.2906 | 0.1700 | 170.7 |
| 12 | 0.5391 | 0.5571 | 0.3446 | 0.2082 | 168.2 |
| 13 | 0.5402 | 0.5556 | 0.4095 | 0.2575 | 168.7 |
| 14 | 0.5424 | 0.5669 | 0.3653 | 0.2235 | 147.4 |
| 15 | 0.5332 | 0.5463 | 0.3816 | 0.2358 | 137.7 |

### Key curve observations
1. **Slow, shallow training loss descent:** Training loss drops from `0.6424` (Epoch 1) to `0.5688` (Epoch 4), and then decreases very slowly over the remaining 11 epochs to `0.5332` (Epoch 15).
2. **Volatile validation loss:** Validation loss tracks in the `0.546–0.648` band without diverging upward, but exhibits sharp spikes at Epochs 3–4 (`0.645`) and Epoch 11 (`0.610`). With `batch_size=2` (`201` gradient updates per epoch) and a fixed learning rate of `0.001`, BatchNorm statistics and weight updates oscillate around local minima.

---

## 3. Validation Dice and IoU Progression

- **High epoch-to-epoch instability:** Validation Dice drops from `0.3617` (Epoch 1) down to `0.1453` (Epoch 3), jumps to `0.4028` (Epoch 4), peaks at **`0.4324`** (IoU **`0.2758`**) at Epoch 8, drops to `0.2906` at Epoch 11, recovers to `0.4095` at Epoch 13, and finishes at `0.3816` at Epoch 15.
- **Disconnect between combined loss and thresholded Dice:** Notice that Epoch 15 actually achieves the lowest validation loss (`0.5463`), yet its validation Dice (`0.3816`) is `0.0508` lower than Epoch 8 (`0.4324`, validation loss `0.5729`). Because `BCEWithLogitsLoss` dominates background pixel calibration (~85% of pixels are background), lowering BCE slightly by shrinking foreground confidence can reduce total loss while hurting thresholded foreground overlap.
- **Need for LR scheduling:** Maintaining a fixed `lr=0.001` after Epoch 8 prevents the model from settling into the high-Dice basin discovered at Epoch 8.

---

## 4. Damage versus Background Pixel Proportions

Quantitative analysis of the ground-truth masks and baseline predictions reveals significant class imbalance and prediction bias:

- **Full archive (583 native-resolution masks):**
  - Total pixels: `10,706,940,000`
  - Damage pixels (`>= 128`): `1,626,206,511` (**`15.19%`**)
  - Background pixels (`< 128`): `9,080,733,489` (**`84.81%`**)
- **Included validation split (81 masks resized to `128 × 128`):**
  - Overall ground-truth damage proportion: **`15.62%`**
  - Per-image ground-truth damage proportion varies wildly:
    - Minimum: **`0.04%`** (tiny isolated crack/defect)
    - Median: **`9.25%`**
    - Maximum: **`77.69%`** (broad damaged pavement surface)
- **Baseline model predictions on validation split (Epoch 8 checkpoint at threshold `0.50`):**
  - Overall predicted damage proportion: **`19.22%`** (over-predicting total damage area by ~23% relative to the `15.62%` ground truth).
  - Per-image predicted damage proportion:
    - Minimum: **`0.00%`**
    - Median: **`20.72%`** (more than double the ground-truth median of `9.25%`)
    - Maximum: **`33.05%`**
  - **Precision vs Recall imbalance:** Global validation Precision is **`0.3919`** while Recall is **`0.4821`**. False positives outnumber true positives (`FP > TP`), dragging down both Precision and Dice.

---

## 5. Visual Diagnosis of Predictions (`validation_predictions.png`)

Inspecting the 4 validation examples in `artifacts/unet_baseline/samples/validation_predictions.png` alongside their quantitative per-sample scores explains the failure modes clearly:

1. **Row 0 (`Val/1004.jpg` — Thin diagonal crack):**
   - **Ground truth:** Thin branching crack covering only **`2.43%`** of the image.
   - **Prediction:** A huge central oval blob covering **`27.06%`** of the image (`Dice = 0.1031`, `IoU = 0.0543`).
   - **Diagnosis:** The model cannot resolve the narrow crack at `128 × 128` and instead fires broadly over the entire central pavement surface (**predictions are far too large / diffuse** on thin-crack images).
2. **Row 1 (`Val/1010.jpg` — Broad worn/damaged slab area):**
   - **Ground truth:** Large contiguous lower-half polygon covering **`45.79%`** of the image plus an upper strip.
   - **Prediction:** Rounded central blob covering **`17.16%`** (`Dice = 0.4171`, `IoU = 0.2635`).
   - **Diagnosis:** The model captures the center of the damaged area but misses the outer boundaries and upper strip (**missing damage regions near edges**).
3. **Row 2 (`Val/1019.jpg` — Intact tiled footpath with utility cover and tactile paving):**
   - **Ground truth:** Tiny defect near the top edge covering only **`0.13%`** of pixels.
   - **Prediction:** Large fragmented blob covering **`20.72%`** over the regular paving tiles and manhole cover (`Dice = 0.0000`, `IoU = 0.0000`).
   - **Diagnosis:** Normal tile joints and utility cover borders are confused with cracks/damage at low resolution (**high false-positive rate on structured pavement textures, with grid-like fragmentation**).
4. **Row 3 (`Val/1022.jpg` — Pronounced vertical crack/trench):**
   - **Ground truth:** Vertical damage region covering **`21.02%`** of the frame.
   - **Prediction:** Well-aligned vertical region covering **`28.45%`** (`Dice = 0.6949`, `IoU = 0.5324`), with minor dilations around the boundary.
   - **Diagnosis:** When damage has high contrast and occupies ~20–30% of the frame, the baseline U-Net localizes it reasonably well, though boundaries remain overly smooth.

**Summary of visual errors:** The baseline model acts like a coarse spatial prior that predicts a **`15%–30%` central blob** on almost every image (median predicted area `20.72%`, max `33.05%`). Consequently:
- On **thin cracks or small defects** (`< 5%` GT area), predictions are **grossly oversized blobs** with massive false positives.
- On **textured paving blocks/manhole covers**, tile seams trigger **fragmented false-positive islands**.
- On **very large damage zones** (`> 45%` GT area), predictions are **too small and overly rounded**, truncating image-border damage.

---

## 6. Underfitting vs. Overfitting Assessment

The baseline model is **primarily underfitting (representation- and optimization-limited), not overfitting**:

- **Evidence against classic overfitting:**
  - Training loss (`0.5332` at Epoch 15) and validation loss (`0.5463` at Epoch 15) remain very close (`gap = 0.0131`).
  - Training loss itself plateaus at a relatively high value (`0.533`), indicating that the model is unable to fit the training masks tightly at `128 × 128`.
- **Causes of underfitting and spatial bias:**
  1. **Severe resolution bottleneck (`128 × 128`):** Downsampling `6000 × 4000` images by `46.9×` horizontally and `31.25×` vertically smears 5–15 pixel wide cracks into sub-pixel low-contrast gray blurs. Unable to see fine crack features, the network minimizes loss by predicting a generic central probability blob matching the average dataset foreground rate (~15–20%).
  2. **Per-image Dice loss vs. Global Dataset Dice metric mismatch:** In `train_unet.py`, `dice_loss` computes the soft Dice score **per image** (`dims = (2, 3)`) and averages across the batch (`1.0 - score.mean()`). However, validation `dice_coefficient` computes **global micro-averaged Dice** over the entire validation tensor `(81, 1, H, W)`. When an image has only `0.1%–2%` ground-truth damage, per-image soft Dice heavily penalizes missing those few pixels, encouraging the network to hedge by predicting broad fuzzy foreground regions across every image.
  3. **Unscheduled learning rate with batch size 2:** Constant `lr=0.001` with 2-sample batches introduces gradient noise that prevents fine boundary convergence.

---

## 7. Impact of `128 × 128` Input Resolution

- `74.8%` of the dataset (`436 / 583` images) has native dimensions of `6000 × 4000` (24 megapixels, `3:2` aspect ratio), while the remaining images are `1500 × 1000` or `1800 × 1200` (also `3:2`).
- Resizing to `128 × 128` (`16,384` pixels) discards **99.93%** of the pixels in a `6000 × 4000` image and slightly compresses the `3:2` horizontal aspect ratio into `1:1`.
- More critically, after 4 max-pooling layers in U-Net (`stride=2` each), the bottleneck spatial map at `128 × 128` is only **`8 × 8`**.
- Increasing the input resolution to **`256 × 256`** quadruples the spatial pixel density (`65,536` pixels) and doubles the linear resolution along both axes (giving a `16 × 16` bottleneck), preserving thin cracks, tile edges, and boundary contours while remaining tractable on CPU.

---

## 8. Computational Bottleneck & ZIP Loading Analysis

To determine why the 15-epoch `128 × 128` baseline required **`40 minutes 48 seconds` (`2447.8s`, or `161.7s/epoch`)**, we benchmarked data I/O against PyTorch U-Net forward/backward computation on this machine:

| Operation | Measured Time | Per-Epoch Cost (402 train + 81 val = 483 samples) | Share of Baseline 161.7s Epoch |
|---|---:|---:|---:|
| **ZIP decompression + 6000×4000 JPEG/PNG decode + resize** | `~0.29s / pair` | **`~140–142s`** | **`~87.5%`** |
| **U-Net forward + backward (`128 × 128`, batch=2)** | `0.097s / train batch` | **`~20–21s`** | **`~12.5%`** |
| **U-Net forward + backward (`256 × 256`, batch=2)** | `0.380s / train batch` | **`~78–82s`** | — |

### Critical finding on ZIP loading
- **Yes — reading and decoding raw `6000 × 4000` images from the 4.2 GB ZIP archive on every single epoch is the overwhelming bottleneck (`~87.5%` of total training time).**
- In each of the 15 epochs, Python decompresses and decodes 483 multi-megapixel images (~3.5 GB of uncompressed pixel data per epoch, or >52 GB over 15 epochs) only to immediately downsample them to `128 × 128` or `256 × 256`.
- **Actionable efficiency solution (Part D):** If we cache the orientation-corrected, resized image/mask pairs (or extracted/pre-resized tensors/PNGs) in a local Git-ignored cache directory under `artifacts/` (e.g., `artifacts/dataset_cache/`) while keeping the original ZIP untouched:
  - Data loading time per epoch drops from **~142 seconds to < 2 seconds**.
  - A 15-epoch `128 × 128` training run drops from **~41 minutes to ~5.5 minutes**.
  - A 15-epoch `256 × 256` training run drops from **~55 minutes (if read from ZIP) to ~20 minutes**.

---

## 9. Recommendations for Controlled Phase 1 Experiments

Based on this diagnosis, Phase 1 experiments will proceed in four controlled steps under `artifacts/unet_experiments/`:

1. **Experiment 1 — Higher Input Resolution (`256 × 256`):**
   - Increase input size from `128 × 128` to `256 × 256` (with nearest-neighbour mask interpolation and bilinear image interpolation) while keeping the U-Net architecture (`base_channels=16`), optimizer (`Adam`, `lr=0.001`), loss (`0.5 BCE + 0.5 Dice`), and seed (`42`) identical to the baseline.
   - Use a deterministic local disk cache of resized benchmark splits (under Git-ignored `artifacts/dataset_cache/`) so the original ZIP is untouched and the 140s/epoch ZIP decompression bottleneck is eliminated after the one-time build.
2. **Experiment 2 — Training Improvements (Augmentation + Scheduler + Early Stopping):**
   - Footpath images are captured from top-down/oblique pedestrian perspectives where horizontal flips and small-angle rotations (`±10°`, using bilinear for image and nearest-neighbour for mask) are domain-valid, whereas vertical flips are avoided for oblique views (inverting ground/curb perspective).
   - Add `ReduceLROnPlateau` (monitoring validation Dice, `mode="max"`, `factor=0.5`, `patience=2`) to damp the post-Epoch-8 oscillations observed in the baseline, plus early stopping (`patience=5`) with `epochs=20`.
3. **Experiment 3 — Controlled Loss Function Comparison:**
   - Compare three controlled loss configurations on the improved setup:
     1. Equal-weight `0.5 * BCE + 0.5 * Dice` (baseline formula).
     2. Dice-focused loss (`0.2 * BCE + 0.8 * Dice` or pure/batch-level soft Dice) to directly optimize foreground overlap and reduce diffuse background/foreground hedging.
     3. Pos-weighted or modestly reweighted `BCE + Dice` (`0.35 * BCE + 0.65 * Dice` or batch-global soft Dice) to align the training objective with the global validation Dice metric.
4. **Experiment 4 — Validation Threshold Selection:**
   - Evaluate candidate probability thresholds `{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}` on the validation set predictions of the best checkpoint.
   - Select the optimal threshold using validation Dice only, reporting both the fixed `0.50` result and the selected threshold result.

---

## 10. Post-Experiment Verification Summary

All four recommendations above were empirically validated in Phase 1 (see `docs/MODEL_EXPERIMENTS.md` for full logs):

- **Caching verification:** On-disk resized PNG caching (`artifacts/dataset_cache/`) cut `256 × 256` epoch runtime to **`73.1–87.1s/epoch`**, making `256 × 256` training faster than the `161.7s/epoch` `128 × 128` ZIP-loaded baseline.
- **Resolution & training stabilization (`exp2_train_improvements`):** Moving to `256 × 256` with `batch_size=4`, `±10°` rotation + horizontal flip, and `ReduceLROnPlateau` lifted validation Dice at `t=0.50` from `0.4324` to **`0.4775`** (IoU `0.2758` -> **`0.3137`**, Precision `0.3919` -> **`0.4578`**), and eliminated false positives on the tiled footpath validation sample (`Val/1019.jpg`).
- **Loss & threshold verification (`exp3b_modest_weight` & `exp2_train_improvements`):** Modest reweighting (`0.35 BCE + 0.65 batch Dice` in `exp3b_modest_weight`) achieved the highest validation Dice at the default `0.50` threshold (**`0.4845`**, IoU **`0.3197`**), while threshold selection (`t* = 0.30` on `exp2_train_improvements`) achieved the highest overall validation Dice (**`0.4888`**, IoU **`0.3235`**).
