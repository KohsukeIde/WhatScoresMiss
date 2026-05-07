# Data

No raw datasets are redistributed in this repository. The code uses existing
datasets under their original terms. Users must obtain each dataset from its
official source and comply with the applicable access, citation, and use terms.

The expected paths are configured through environment variables. Directory
layouts below are illustrative and should be adapted to the official release
format and the external model codebase used for Level 2 diagnostics.

## Dataset Summary

| dataset | official source | terms pointer | purpose in paper | environment variable | preprocessing note |
|---|---|---|---|---|---|
| ScanNet | `https://www.scan-net.org/ScanNet/` | ScanNet Terms of Use linked from the official site | scene-level support and reported-score diagnostics | `SCANNET_ROOT` | run the preprocessing expected by the external scene model code |
| ARKitScenes | `https://github.com/apple/ARKitScenes` | license/terms files in the official repository | Concerto target-swap and coordinate-closure diagnostics | `ARKITSCENES_ROOT` | download requested splits and build any image/point feature caches required by the external code |
| ScanNet++ | `https://kaldir.vc.in.tum.de/scannetpp/` | `https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf` | Concerto target-swap and coordinate-closure diagnostics | `SCANNETPP_ROOT` | run the release-specific scene preprocessing required by the external code |
| S3DIS | `http://3dsemantics.stanford.edu/` | official dataset access page and accompanying terms | Concerto target-swap and coordinate-closure diagnostics | `S3DIS_ROOT` | prepare area folders or preprocessed blocks expected by the external scene code |
| HM3D | `https://aihabitat.org/datasets/hm3d/` | terms linked from the HM3D/Habitat-Matterport release pages | Concerto target-swap and coordinate-closure diagnostics | `HM3D_ROOT` | download the required split files and build any feature caches required by the external code |
| Structured3D | `https://structured3d-dataset.org/` | Structured3D Terms of Use linked from the official site | Concerto target-swap and coordinate-closure diagnostics | `STRUCTURED3D_ROOT` | download the required release files and build any feature caches required by the external code |
| ScanObjectNN | `https://github.com/hkust-vgd/scanobjectnn` | official repository and dataset release terms | object classification, checkpoint reporting, and PointGPT mask/order diagnostics | `SCANOBJECTNN_ROOT` | use the H5 layout expected by PointGPT, Point-MAE, and PCP-MAE code |
| ShapeNetPart | `https://www.shapenet.org/` | ShapeNet terms at `https://www.shapenet.org/terms` | object part-segmentation support diagnostics | `SHAPENETPART_ROOT` | use the standard ShapeNetPart part-annotation layout expected by the external object code |

## Expected Local Layouts

### ScanNet

```text
$SCANNET_ROOT/
  scans/
  meta_data/
  preprocessed/        # optional; model-code specific
```

### ARKitScenes

```text
$ARKITSCENES_ROOT/
  threedod/
  upsampling/
  raw/
  preprocessed/        # optional; model-code specific
  feature_cache/       # optional
```

### ScanNet++

```text
$SCANNETPP_ROOT/
  data/
  splits/
  preprocessed/        # optional; model-code specific
```

### S3DIS

```text
$S3DIS_ROOT/
  Area_1/
  Area_2/
  Area_3/
  Area_4/
  Area_5/
  Area_6/
  preprocessed/        # optional; model-code specific
```

### HM3D

```text
$HM3D_ROOT/
  train/
  val/
  minival/
  semantic_annots/     # optional, depending on release variant
  feature_cache/       # optional
```

### Structured3D

```text
$STRUCTURED3D_ROOT/
  scene_00000/
  scene_00001/
  ...
  splits/
  feature_cache/       # optional
```

### ScanObjectNN

```text
$SCANOBJECTNN_ROOT/
  h5_files/
    main_split/
    main_split_nobg/
    main_split_nobg_augmentedrot_scale75/
```

For `PB_T50_RS`, use the method-compatible hardest-split H5 files expected by
the external PointGPT, Point-MAE, and PCP-MAE code.

### ShapeNetPart

```text
$SHAPENETPART_ROOT/
  02691156/
  02773838/
  ...
  train_test_split/
  synsetoffset2category.txt
```

## Preprocessing

Level 0 and Level 1 do not require datasets. Level 2 wrappers assume that the
external codebase used for each model can find its usual preprocessed data. If
preprocessing is required, run the official or model-code-specific preprocessing
pipeline for the corresponding dataset before launching full diagnostics.
