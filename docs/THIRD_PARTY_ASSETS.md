# Third-Party Assets

This repository does not claim ownership of external datasets, checkpoints, or
third-party model code. Users must obtain third-party assets from their official
sources and comply with the corresponding terms.

## Datasets

- ScanNet
- ARKitScenes
- ScanNet++
- S3DIS
- HM3D
- Structured3D
- ScanObjectNN
- ShapeNetPart

See `docs/DATA.md` for path variables and usage notes.

## Model Families and Codebases

- Concerto: scene-level cross-modal self-supervised learning diagnostics.
- Sonata: scene-level self-supervised comparator diagnostics.
- Utonia: scene-level self-supervised comparator diagnostics.
- PTv3: supervised scene control diagnostics.
- PointGPT: object-level autoregressive point-cloud pretraining diagnostics.
- Point-MAE: object-level masked autoencoding baseline diagnostics.
- PCP-MAE: side-channel-aware masked autoencoding diagnostics.

Where known, use official project releases for pretrained checkpoints and model
implementations. For any codebase or checkpoint not redistributed here, set the
environment variables described in `docs/CHECKPOINTS.md`.

## Papers

The paper relies on established datasets and model families. Cite the original
dataset and model papers when using their assets. Verify the final citation list
before camera-ready submission.
