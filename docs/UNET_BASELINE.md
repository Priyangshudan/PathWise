# U-Net CPU baseline

## Model and data

`cv/models/unet.py` implements a four-level U-Net with two-convolution blocks, max-pooling, a 256-channel bottleneck, transpose-convolution upsampling, and skip connections. The default widths are 16, 32, 64, 128, and 256. It accepts RGB float images in `[0,1]` with shape `N×3×128×128` and returns one-channel raw logits of the same spatial size. No pretrained weights are used.

The training script reads the original ZIP through `FootpathDamageDataset` and filters with `cv/data/experiment_manifest.csv`. Quarantined rows are excluded. Training uses synchronized random horizontal flips and resizing; validation and test use resizing only. Masks use nearest-neighbour resizing and remain binary.

## Initial training configuration

- CPU, seed 42 (Python, NumPy, PyTorch, and shuffle generator)
- 128×128 input, batch size 2, 15 epochs
- Adam, learning rate 0.001
- Loss: equal-weight mean of `BCEWithLogitsLoss` and soft Dice loss; Dice smoothing handles empty masks
- Default DataLoader workers: 0 (configurable)
- Validation Dice and IoU at probability threshold 0.5; test metrics also include precision, recall, and F1

## Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s cv\tests -v
.\.venv\Scripts\python.exe cv\training\train_unet.py
```

Paths and training values can be overridden with command-line options; run `.\.venv\Scripts\python.exe cv\training\train_unet.py --help` for the list. The default run uses the existing dataset ZIP and derived manifest.

After training, evaluate the best checkpoint on the held-out test split:

```powershell
.\.venv\Scripts\python.exe cv\training\train_unet.py --evaluate-checkpoint artifacts\unet_baseline\best_model.pth
```

## Outputs

All outputs go to `artifacts/unet_baseline/` (ignored by Git):

- `best_model.pth`: best validation-Dice model, optimizer, epoch, model/training configuration, and seed
- `history.csv` and `history.json`: epoch losses, validation Dice/IoU, and epoch timing
- `training_curves.png`: training/validation loss and validation Dice/IoU curves
- `samples/validation_predictions.png`: original resized image, ground truth, prediction, and overlay panels
- `training_summary.json`: best and final validation metrics plus runtime
- `test_metrics.json`: written by the optional checkpoint evaluation command
