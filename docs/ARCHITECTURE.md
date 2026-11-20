# PathWise architecture

## Current milestone

The initial implementation is a Python computer vision pipeline for auditing and preparing footpath damage segmentation data. PyTorch is planned for dataset interfaces and model development. OpenCV is planned for image and video processing; it is not yet installed or implemented.

## Planned components

- **Android application:** future mobile client; placeholder directory only.
- **Python computer vision pipeline:** initial implementation; dataset audit and loading only at this milestone.
- **PyTorch:** planned model development and training; no model is implemented or trained.
- **OpenCV:** planned image/video processing; not implemented.
- **FastAPI:** planned future local inference API; not implemented.
- **Room/SQLite:** planned future local Android report storage; not implemented.
- **MapLibre:** planned future map interface; not implemented.

The intended first experiment is binary semantic segmentation of footpath damage. Dataset encoding must be verified before training code is added.
