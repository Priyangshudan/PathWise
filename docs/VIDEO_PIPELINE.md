# PathWise — Offline Video Processing and Segmentation Pipeline (`docs/VIDEO_PIPELINE.md`)

## 1. Overview and Architecture

The PathWise offline video pipeline (`cv/video/`) processes local pedestrian footpath recordings using OpenCV and the trained PyTorch U-Net checkpoint (`artifacts/unet_experiments/exp2_train_improvements/best_model.pth`). It is designed to run efficiently on a CPU-only Windows machine with 16 GB RAM without loading entire video streams into memory.

```text
Input Video (.mp4 / .avi / .mov / .mkv)
   │
   ▼
cv/video/video_reader.py (VideoReader)
   ├── Streams frames sequentially via cv2.VideoCapture
   └── Extracts FPS, width, height, frame count & calculates timestamps (frame_index / fps)
   │
   ▼
cv/video/frame_sampler.py (FrameSampler)
   ├── Configurable time-based sampling (default: every 1.0 second)
   └── Optional consecutive near-duplicate suppression via 64-bit perceptual dHash
   │
   ▼
cv/video/video_inference.py (VideoSegmentationModel & annotate_frame)
   ├── Loads checkpoint once per run (default: exp2_train_improvements/best_model.pth)
   ├── Exact training preprocessing: BGR -> RGB PIL -> 256×256 BILINEAR -> float32 [0,1]
   ├── U-Net forward pass under torch.inference_mode() -> sigmoid -> threshold (default: 0.30)
   ├── Upsamples binary mask to native video resolution via NEAREST interpolation
   └── Renders translucent red fill, yellow-orange boundary contour & compact top-left HUD
   │
   ▼
cv/video/process_video.py (run_video_pipeline / CLI)
   ├── Writes annotated output video (<stem>_annotated.mp4 / .avi) at native resolution
   ├── Writes per-frame metrics to frame_results.csv
   └── Writes run metadata and aggregate statistics to summary.json
```

---

## 2. Frame Ingestion and Timestamp Handling (`cv/video/video_reader.py`)

- **Sequential streaming:** `VideoReader` wraps `cv2.VideoCapture` as a context manager (`with VideoReader(path) as reader:`) that yields one `VideoFrame(frame_index, timestamp_seconds, image_bgr)` at a time and guarantees `cap.release()` on exit or exception.
- **Supported formats:** `.mp4` (primary), `.avi`, `.mov`, `.mkv`, `.webm`, and `.m4v`. Missing paths, non-file paths, unsupported extensions, or corrupt files with zero readable frames raise descriptive `FileNotFoundError` or `ValueError` exceptions.
- **Timestamp calculation & metadata fallbacks:**
  - When the container reports a valid frame rate (`0.1 <= fps <= 240.0`), `fps_reliable` is set to `True` and each frame's timestamp is computed deterministically as:
    `timestamp_seconds = frame_index / fps`
  - If container metadata reports a missing or invalid FPS (`<= 0` or non-finite), `VideoReader` falls back to `default_fps = 30.0` and records `fps_reliable = False` in `summary.json`.
  - If frame dimensions (`width`, `height`) are missing from container headers, the reader probes the first frame to infer dimensions and buffers it so iteration still starts cleanly at `frame_index = 0`.

---

## 3. Time-Based Frame Sampling & Near-Duplicate Suppression (`cv/video/frame_sampler.py`)

- **Time-based sampling:** Adjacent frames in 30–60 FPS walking videos are highly redundant. `FrameSampler(sample_interval_seconds=1.0)` samples the first frame (`t = 0.0s`) and then advances the target timestamp in increments of `sample_interval_seconds` (configurable via `--sample-interval`; setting `0` processes every frame). Original `frame_index` and `timestamp_seconds` from the source video are preserved.
- **Consecutive near-duplicate suppression (`--skip-similar-frames`):**
  - When enabled, each sampled frame is converted to a 64-bit horizontal difference hash (`compute_dhash`: grayscale conversion -> `9 × 8` `cv2.INTER_AREA` resize -> horizontal adjacent pixel comparisons `col[c+1] > col[c]`).
  - The bitwise Hamming distance (`hamming_distance`) is computed against the last non-skipped sampled frame. If `hamming_distance <= max_hamming_distance` (default `4` out of 64 bits), the frame is flagged `skipped_as_duplicate = True` and reuses the previous frame's segmentation mask without running redundant neural network inference.
  - **Important scope note:** This perceptual hash comparison is purely visual and only suppresses redundant consecutive frames when the camera pauses. It does **not** track physical defects across camera motion, nor does it suppress visually distinct frames that happen to have similar damage percentages.

---

## 4. Segmentation Model Inference & Threshold Selection (`cv/video/video_inference.py`)

- **Single model load:** `VideoSegmentationModel` loads the checkpoint once at startup onto `cpu` (default) and sets `model.eval()`.
- **Exact preprocessing reproduction:**
  1. Converts the OpenCV `BGR` frame (`uint8`, `H × W × 3`) to `RGB` and wraps it in a `PIL.Image`.
  2. Resizes to `(input_size, input_size)` (`256 × 256` for `exp2_train_improvements`, read automatically from `checkpoint["training_config"]["input_size"]`) using **`Image.Resampling.BILINEAR`**.
  3. Normalizes pixel intensities to `float32` in `[0.0, 1.0]` (`/ 255.0`) and rearranges axes to `(1, 3, H, W)`.
- **Logit thresholding & mask upsampling:**
  - Executes the forward pass inside `torch.inference_mode()`, applies `torch.sigmoid()` to the raw 1-channel logits, and thresholds at `threshold`.
  - Defaults to **`0.30`**, the validation-selected optimal threshold (`t*`) for `exp2_train_improvements` (which achieved **`0.4888` validation Dice** and **`0.5331` held-out test Global Dice**).
  - Resizes the binary `{0, 1}` prediction mask back to the original video resolution `(W, H)` using **nearest-neighbour interpolation (`cv2.INTER_NEAREST`)** so binary class labels remain discrete.

---

## 5. Annotated Video and Structured Outputs (`cv/video/process_video.py`)

Running the pipeline produces three outputs in `--output-dir`:

1. **Annotated Video (`<input_stem>_annotated.mp4` or `.avi`):**
   - Preserves the native width and height of the input video.
   - Highlights predicted footpath damage with a translucent red overlay (`45%` blend) and a `2px` yellow-orange contour outline (`cv2.findContours`) around damaged regions.
   - Draws a compact top-left status bar showing `Frame: <idx> | Time: <sec>s | Thr: <thr> | Damage: <pct>%`.
   - Uses `mp4v` codec for `.mp4` output by default, with automatic fallback to `avc1` (`.mp4`), `XVID` (`.avi`), or `MJPG` (`.avi`) if a codec is unavailable in the local OpenCV build.
2. **Per-Frame CSV (`frame_results.csv`):**
   - Columns: `source_video`, `frame_index`, `timestamp_seconds`, `probability_threshold`, `damaged_pixel_count`, `damaged_percentage`, `skipped_as_duplicate`, `hamming_distance_from_kept`, `inference_seconds`, `processing_seconds`.
3. **Run Summary (`summary.json`):**
   - Records `source_video`, `video_properties` (`width`, `height`, `fps`, `fps_reliable`, `reported_total_frames`, `estimated_duration_seconds`), `pipeline_configuration`, `frame_counts` (`total_frames`, `sampled_frames`, `processed_frames`, `skipped_near_duplicate_frames`), `timing` (`average_inference_seconds`, `total_processing_seconds`, `sampled_frames_per_second`), `predicted_damage_area` (`average_damaged_pixels_processed`, `average_damaged_percentage_processed`, `average_damaged_percentage_all_sampled`, `max_damaged_percentage_sampled`, plus a non-physical-measurement disclaimer), and output file paths.

---

## 6. Command-Line Usage

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m cv.video.process_video `
  --input "path\to\footpath_video.mp4" `
  --output-dir artifacts\video_demo `
  --checkpoint artifacts\unet_experiments\exp2_train_improvements\best_model.pth `
  --threshold 0.30 `
  --sample-interval 1.0 `
  --device cpu
```

To enable consecutive near-duplicate frame suppression (recommended when the camera pauses while recording):

```powershell
.\.venv\Scripts\python.exe -m cv.video.process_video `
  --input "path\to\footpath_video.mp4" `
  --output-dir artifacts\video_demo `
  --checkpoint artifacts\unet_experiments\exp2_train_improvements\best_model.pth `
  --threshold 0.30 `
  --sample-interval 1.0 `
  --skip-similar-frames `
  --device cpu
```

---

## 7. Known Limitations

1. **Image-plane pixel area is not a physical measurement:** `damaged_pixel_count` and `damaged_percentage` measure 2D perspective-projected image coverage only. Camera height, tilt angle, and lens field-of-view alter apparent pixel area; no real-world crack width, depth, or structural severity is inferred.
2. **No multi-frame physical defect tracking (SLAM / optical flow):** Frame-by-frame segmentation processes each sampled timestamp independently. Near-duplicate suppression only skips consecutive visually identical frames when the camera is stationary; a walking video moving past a single long crack will segment that crack in multiple successive sampled frames.
3. **Motion blur and lighting transitions:** Rapid smartphone panning, strong shadows, wet pavement reflections, or heavy compression artifacts can reduce segmentation precision compared to still photographs.
4. **Square `256 × 256` model input:** Native `16:9` or `9:16` smartphone video frames are bilinearly resized to `256 × 256` for U-Net inference and nearest-neighbour resized back to native resolution, which compresses the aspect ratio during the forward pass.
