# Third-Party Assets

This repository does not claim ownership of external datasets, checkpoints, or
third-party model code. Users must obtain third-party assets from their official
sources and comply with the corresponding terms.

## Datasets

Dataset sources, access notes, and expected local path variables are listed in
`docs/DATA.md`.

## Model Families and Codebases

| model/codebase | role in this repository | official project or paper source |
|---|---|---|
| Concerto | scene-level cross-modal self-supervised diagnostics | `https://github.com/Pointcept/Concerto`, `https://pointcept.github.io/Concerto/`, `https://arxiv.org/abs/2510.23607` |
| Sonata | scene-level self-supervised comparator diagnostics | `https://github.com/facebookresearch/sonata`, `https://arxiv.org/abs/2503.16429` |
| Utonia | scene-level self-supervised comparator diagnostics | `https://arxiv.org/abs/2603.03283` |
| PTv3 | supervised scene-control diagnostics | `https://github.com/Pointcept/PointTransformerV3`, `https://arxiv.org/abs/2312.10035` |
| PointGPT | object-level autoregressive point-cloud pretraining diagnostics | `https://github.com/CGuangyan-BIT/PointGPT`, `https://arxiv.org/abs/2305.11487` |
| Point-MAE | object-level masked-autoencoding baseline diagnostics | `https://github.com/Pang-Yatian/Point-MAE`, `https://arxiv.org/abs/2203.06604` |
| PCP-MAE | side-channel-aware masked-autoencoding diagnostics | `https://github.com/aHapBean/PCP-MAE`, `https://arxiv.org/abs/2408.08753` |

## Checkpoints and External Code

No large checkpoints or external model implementations are bundled here. Level 2
wrappers validate local paths and document diagnostic entry points. To execute
full diagnostics, install the corresponding official model codebases, obtain the
appropriate checkpoints under their release terms, and set the environment
variables listed in `docs/CHECKPOINTS.md`.

## Citation Responsibility

When using third-party datasets, checkpoints, or model code, cite the original
dataset/model papers and follow the official license or terms pages. This
repository provides a diagnostic package for the evaluation study.
