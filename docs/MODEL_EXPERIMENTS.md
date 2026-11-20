# PathWise — Phase 1 U-Net Controlled Model Experiments (`docs/MODEL_EXPERIMENTS.md`)

## 1. Experimental Protocol and Setup

All Phase 1 experiments were executed under strict controls to measure incremental improvements over the original U-Net baseline (`artifacts/unet_baseline/best_model.pth`):

- **Dataset & Manifest:** `Datasets/Indian Footpath Damage Segmentation Dataset.zip` filtered via `cv/data/experiment_manifest.csv` (**402 train**, **81 validation**, **82 held-out test**; all 18 duplicate-image rows across `DG01`–`DG09` excluded).
- **Test-Set Isolation:** The held-out test split (`82` samples) was **not** accessed or evaluated during model development, loss comparisons, or threshold selection.
- **Baseline Preservation:** `artifacts/unet_baseline/best_model.pth` and its companion files were preserved without modification. All new checkpoints and logs were saved under `artifacts/unet_experiments/<experiment_name>/`.
- **Reproducibility:** Fixed random seed `42` (`seed_everything(42)`) across Python, NumPy, PyTorch, and the DataLoader sampler.
- **Architecture:** Compact 4-level U-Net (`cv/models/unet.py`, `base_channels=16`, encoder widths `16, 32, 64, 128`, bottleneck `256`, `out_channels=1` raw logits).
- **Computational Efficiency (Part D):** On-disk caching of orientation-corrected, resized PNG pairs (`artifacts/dataset_cache/`, ignored by Git) eliminated repeated per-epoch decompression and decoding of `6000 × 4000` JPEGs/PNGs from the 4.2 GB ZIP archive. This reduced `256 × 256` epoch runtime to **`73–87 seconds/epoch`** on CPU (`num_workers=0`), making `256 × 256` training **35%–46% faster** in wall-clock time than the original `128 × 128` baseline (`161.7 seconds/epoch`).

---

## 2. Master Comparison Table (Validation Split)

Metrics are global binary segmentation scores on the `81` included validation samples. Both the **fixed `0.50` probability threshold** and the **validation-selected optimal threshold (`t*`)** from `{0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}` are reported.

| Experiment Name | Input Size | Batch | Epochs (Best / Total) | Augmentations | LR & Scheduler | Loss Function | Val Dice (`t=0.50`) | Val IoU (`t=0.50`) | Val Prec (`t=0.50`) | Val Rec (`t=0.50`) | Selected `t*` | Val Dice (`t*`) | Val IoU (`t*`) | Val Prec (`t*`) | Val Rec (`t*`) | Total Runtime (Avg/Ep) |
|---|---:|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`unet_baseline`** *(Reference)* | `128×128` | 2 | 8 / 15 | H-flip (`0.5`) | `0.001` (Fixed) | `0.50 BCE + 0.50 sample Dice` | `0.4324` | `0.2758` | `0.3919` | `0.4821` | `0.40` | `0.4345` | `0.2775` | `0.3665` | `0.5333` | `2447.8s` / `40m 48s` (`161.7s/ep`) |
| **`exp1_res256`** *(Exp 1: Resolution)* | `256×256` | 2 | 4 / 15 | H-flip (`0.5`) | `0.001` (Fixed) | `0.50 BCE + 0.50 sample Dice` | `0.4258` | `0.2705` | `0.4329` | `0.4189` | `0.30` | `0.4452` | `0.2864` | `0.3841` | `0.5296` | `1428.5s` / `23m 49s` (`94.6s/ep`*) |
| **`exp2_train_improvements`** *(Exp 2: Training)* | `256×256` | 4 | 17 / 18 | H-flip (`0.5`), Rotation (`±10°`) | `0.001` + `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.50 BCE + 0.50 sample Dice` | `0.4775` | `0.3137` | **`0.4578`** | `0.4991` | `0.30` | **`0.4888`** | **`0.3235`** | `0.4184` | `0.5877` | `1577.9s` / `26m 18s` (`87.1s/ep`) |
| **`exp3a_dice_focused`** *(Exp 3a: Loss)* | `256×256` | 4 | 15 / 18 | H-flip (`0.5`), Rotation (`±10°`) | `0.001` + `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.20 BCE + 0.80 batch Dice` | `0.4737` | `0.3103` | `0.4337` | `0.5217` | `0.30` | `0.4796` | `0.3155` | `0.4119` | `0.5739` | `1456.7s` / `24m 17s` (`80.4s/ep`) |
| **`exp3b_modest_weight`** *(Exp 3b: Loss)* | `256×256` | 4 | 14 / 18 | H-flip (`0.5`), Rotation (`±10°`) | `0.001` + `ReduceLROnPlateau` (`0.5×`, pat=2) | `0.35 BCE + 0.65 batch Dice` | **`0.4845`** | **`0.3197`** | `0.4072` | **`0.5980`** | `0.40` | `0.4866` | `0.3215` | `0.3943` | `0.6352` | `1324.3s` / `22m 04s` (`73.1s/ep`) |

*\*Note on `exp1_res256` runtime: Epoch 1 took `231.4s` because it populated the `256×256` disk cache from the 4.2 GB ZIP archive; Epochs 2–15 averaged `85.5s/epoch`.*

---

## 3. Detailed Experiment Analysis

### Experiment 1: Higher Input Resolution (`exp1_res256`)
- **Configuration:** Increased input dimensions from `128 × 128` to `256 × 256` (bilinear image interpolation, nearest-neighbour mask interpolation). Kept `batch_size=2`, `epochs=15`, `lr=0.001`, horizontal flip (`p=0.5`), and `0.50 BCE + 0.50 sample Dice` identical to the baseline.
- **Artifacts:** `artifacts/unet_experiments/exp1_res256/`
- **Results & Comparison with Baseline:**
  - At fixed threshold `0.50`, validation Precision jumped by **`+0.0410`** (`0.4329` vs `0.3919`), while Recall dropped (`0.4189` vs `0.4821`), yielding Dice `0.4258` and IoU `0.2705` at Epoch 4.
  - At the validation-selected threshold `0.30`, validation Dice improved to **`0.4452`** (`+0.0128` over baseline fixed `0.50` and `+0.0107` over baseline's best threshold `0.4345`) and IoU improved to **`0.2864`** (`+0.0106`).
  - Visually (`exp1_res256/samples/validation_predictions.png`), moving to `256 × 256` stopped the network from predicting a huge oval blob across the thin diagonal crack in `Val/1004.jpg` (Row 0), allowing it to localize the narrow crack line. However, keeping `batch_size=2` and constant `lr=0.001` still caused severe epoch-to-epoch validation oscillations after Epoch 4 (dropping to `0.2667` by Epoch 15).

### Experiment 2: Training Improvements (`exp2_train_improvements`)
- **Configuration:** Built on `256 × 256` resolution with four modest, domain-appropriate training improvements:
  1. **Synchronized small-angle rotation (`±10°`)** using bilinear resampling for RGB images and nearest-neighbour resampling for binary masks, paired with random horizontal flip (`p=0.5`). Vertical flipping was intentionally omitted because many footpath photographs are oblique pedestrian views where upside-down pavement/curb geometry is unnatural.
  2. **Batch size increased from `2` to `4`** to stabilize BatchNorm statistics and reduce gradient variance at `256 × 256` while staying well within 16 GB RAM.
  3. **`ReduceLROnPlateau` learning-rate scheduler** monitoring validation Dice (`mode="max"`, `factor=0.5`, `patience=2`, `min_lr=1e-5`).
  4. **Early stopping (`patience=5`)** over an `18`-epoch budget.
- **Artifacts:** `artifacts/unet_experiments/exp2_train_improvements/`
- **Results & Comparison with Baseline:**
  - Validation Dice climbed steadily from `0.2859` (Epoch 1) to `0.4463` (Epoch 10). When `ReduceLROnPlateau` halved the learning rate to `0.0005` at Epoch 14, validation Dice surged to `0.4653` (Epoch 14) and peaked at **`0.4775`** (IoU **`0.3137`**) at **Epoch 17** (`t=0.50`).
  - Compared to the original baseline at `t=0.50`:
    - **Validation Dice:** `0.4324` -> **`0.4775`** (**`+0.0452`** absolute / **`+10.4%`** relative improvement)
    - **Validation IoU:** `0.2758` -> **`0.3137`** (**`+0.0379`** absolute / **`+13.7%`** relative improvement)
    - **Validation Precision:** `0.3919` -> **`0.4578`** (**`+0.0659`** absolute improvement)
    - **Validation Recall:** `0.4821` -> **`0.4991`** (**`+0.0170`** absolute improvement)
  - With threshold selection (`t* = 0.30`), validation Dice reached **`0.4888`** and IoU reached **`0.3235`** (**`+0.0565`** Dice and **`+0.0477`** IoU above the original `0.50` baseline).
  - Visually (`exp2_train_improvements/samples/validation_predictions.png`), false positives on the intact tiled footpath with the utility cover (Row 2, `Val/1019.jpg`) disappeared completely.

### Experiment 3: Controlled Loss Function Comparison (`exp2` vs `exp3a` vs `exp3b`)
Holding the Experiment 2 training setup fixed (`256 × 256`, `batch=4`, `±10°` rotation + H-flip, `ReduceLROnPlateau`, `seed=42`, 18 epochs), three controlled loss formulations were compared:

1. **Equal-weight BCE + sample soft Dice (`exp2_train_improvements`):** `0.50 * BCE + 0.50 * sample Dice`
   - Best Epoch: `17` | `t=0.50`: Dice **`0.4775`**, IoU **`0.3137`**, Precision **`0.4578`**, Recall `0.4991`
   - Selected `t*=0.30`: Dice **`0.4888`**, IoU **`0.3235`**, Precision `0.4184`, Recall `0.5877`
2. **Dice-focused loss with batch-level soft Dice (`exp3a_dice_focused`):** `0.20 * BCE + 0.80 * batch Dice`
   - Best Epoch: `15` | `t=0.50`: Dice **`0.4737`**, IoU **`0.3103`**, Precision `0.4337`, Recall `0.5217`
   - Selected `t*=0.30`: Dice **`0.4796`**, IoU **`0.3155`**, Precision `0.4119`, Recall `0.5739`
   - *Finding:* Heavily weighting soft Dice (`0.80`) increased Recall at `t=0.50` (`0.5217` vs `0.4991`) and made the predictions less sensitive to high thresholds (`Dice = 0.4503` even at `t=0.70`), but reducing the pixel-wise BCE penalty to `0.20` allowed more background false positives on textured paving tiles.
3. **Modestly reweighted BCE + batch soft Dice (`exp3b_modest_weight`):** `0.35 * BCE + 0.65 * batch Dice`
   - Best Epoch: `14` | `t=0.50`: Dice **`0.4845`**, IoU **`0.3197`**, Precision `0.4072`, Recall **`0.5980`**
   - Selected `t*=0.40`: Dice **`0.4866`**, IoU **`0.3215`**, Precision `0.3943`, Recall `0.6352`
   - *Finding:* A modest `0.35 / 0.65` weighting with batch-level Dice aggregation achieved the **highest validation Dice (`0.4845`) and IoU (`0.3197`) at the default `0.50` threshold**, improving over the baseline by **`+0.0521` Dice** and **`+0.0439` IoU** at `t=0.50`, and shifting the natural probability calibration closer to `0.50` (optimal threshold `0.40` vs `0.30`).

---

## 4. Experiment 4: Validation Threshold Selection

Candidate probability thresholds from `0.30` to `0.70` (step `0.05`) were evaluated on the validation predictions of each best checkpoint (`threshold_sweep.json` in each experiment directory):

| Threshold (`t`) | `unet_baseline` Dice / IoU | `exp1_res256` Dice / IoU | `exp2_train_improvements` Dice / IoU | `exp3a_dice_focused` Dice / IoU | `exp3b_modest_weight` Dice / IoU |
|---:|---:|---:|---:|---:|---:|
| `0.30` | `0.4336` / `0.2768` | **`0.4452`** / **`0.2864`** | **`0.4888`** / **`0.3235`** | **`0.4796`** / **`0.3155`** | `0.4849` / `0.3201` |
| `0.35` | `0.4341` / `0.2772` | `0.4420` / `0.2837` | `0.4875` / `0.3223` | `0.4785` / `0.3145` | `0.4862` / `0.3212` |
| `0.40` | **`0.4345`** / **`0.2775`** | `0.4386` / `0.2809` | `0.4854` / `0.3205` | `0.4771` / `0.3133` | **`0.4866`** / **`0.3215`** |
| `0.45` | `0.4341` / `0.2772` | `0.4328` / `0.2762` | `0.4820` / `0.3175` | `0.4756` / `0.3120` | `0.4858` / `0.3208` |
| **`0.50` (Fixed)** | **`0.4324`** / **`0.2758`** | **`0.4258`** / **`0.2705`** | **`0.4775`** / **`0.3137`** | **`0.4737`** / **`0.3103`** | **`0.4845`** / **`0.3197`** |
| `0.55` | `0.4283` / `0.2725` | `0.4160` / `0.2627` | `0.4714` / `0.3084` | `0.4709` / `0.3079` | `0.4822` / `0.3177` |
| `0.60` | `0.4199` / `0.2658` | `0.4031` / `0.2524` | `0.4626` / `0.3009` | `0.4669` / `0.3045` | `0.4775` / `0.3136` |
| `0.65` | `0.4067` / `0.2552` | `0.3845` / `0.2380` | `0.4496` / `0.2900` | `0.4609` / `0.2994` | `0.4702` / `0.3074` |
| `0.70` | `0.3898` / `0.2421` | `0.3607` / `0.2200` | `0.4332` / `0.2765` | `0.4503` / `0.2906` | `0.4598` / `0.2985` |

### Key Takeaways from Threshold Selection
1. **At the standard fixed `0.50` threshold:** **`exp3b_modest_weight` (`0.35 BCE + 0.65 batch Dice`)** is the best configuration, achieving **Validation Dice `0.4845`** and **Validation IoU `0.3197`** (vs. `0.4324` Dice and `0.2758` IoU for the baseline).
2. **With validation threshold selection (`t*`):** **`exp2_train_improvements` at `t* = 0.30`** achieves the highest overall **Validation Dice `0.4888`** and **Validation IoU `0.3235`**, closely followed by **`exp3b_modest_weight` at `t* = 0.40`** with **Validation Dice `0.4866`** and **Validation IoU `0.3215`**.
3. **Why lower thresholds (`0.30–0.40`) help equal-weight BCE+Dice models:** Because ~85% of pixels in the dataset are background, `0.50 BCE` pulls uncertain boundary and thin-crack logits downward toward background. Lowering the decision threshold to `0.30–0.40` (or using `0.35 BCE + 0.65 batch Dice` in `exp3b`) recovers thin-crack and boundary recall.

---

## 5. Recommended Best Validation Configuration

We identify two complementary top checkpoints depending on whether a fixed `0.50` threshold or a tuned threshold is preferred:

1. **Best Overall Validation Dice (with tuned threshold `t* = 0.30`) & Highest Precision (`t = 0.50`):**
   - **Experiment:** `exp2_train_improvements` (`artifacts/unet_experiments/exp2_train_improvements/best_model.pth`, Epoch 17)
   - **Fixed `t = 0.50`:** Dice **`0.4775`** (`+0.0452` vs baseline), IoU **`0.3137`** (`+0.0379`), Precision **`0.4578`** (`+0.0659`), Recall **`0.4991`** (`+0.0170`)
   - **Selected `t* = 0.30`:** Dice **`0.4888`** (`+0.0565` vs baseline `t=0.50`), IoU **`0.3235`** (`+0.0477`), Precision **`0.4184`**, Recall **`0.5877`**
2. **Best Fixed-Threshold (`t = 0.50`) Validation Dice & Calibration:**
   - **Experiment:** `exp3b_modest_weight` (`artifacts/unet_experiments/exp3b_modest_weight/best_model.pth`, Epoch 14)
   - **Fixed `t = 0.50`:** Dice **`0.4845`** (`+0.0521` vs baseline), IoU **`0.3197`** (`+0.0439`), Precision **`0.4072`** (`+0.0153`), Recall **`0.5980`** (`+0.1159`)
   - **Selected `t* = 0.40`:** Dice **`0.4866`** (`+0.0542` vs baseline `t=0.50`), IoU **`0.3215`** (`+0.0457`), Precision **`0.3943`**, Recall **`0.6352`**

---

## 6. Remaining Limitations

1. **Scratch compact encoder capacity:** The 4-level U-Net (`base_channels=16`, ~1.94M parameters) is trained from scratch on only 402 training images. It still struggles to distinguish some complex surface stains, shadows, and rough aggregate textures from structural footpath damage.
2. **Resolution ceiling (`256 × 256` vs `6000 × 4000`):** While `256 × 256` resolves major cracks far better than `128 × 128`, hairline cracks (`< 10px` wide at `6000 × 4000`) remain near 1 pixel wide at `256 × 256`.
3. **Annotation heterogeneity in source dataset:** As established in the dataset audit, ground-truth masks vary in granularity (some annotate tight crack lines while others annotate broad bounding polygons around worn pavement slabs), placing an upper bound on pixel-wise IoU.
