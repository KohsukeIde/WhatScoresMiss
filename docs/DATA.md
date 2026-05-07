# Data

No raw datasets are redistributed in this repository. The code uses existing
datasets under their original terms. Users must obtain each dataset from its
official source and verify the applicable terms before use.

The expected paths are configured through environment variables. Directory
layouts below are illustrative and should be adapted to the official release
format used by the user.

## Dataset Summary

| dataset | purpose in paper | environment variable | preprocessing | terms note |
|---|---|---|---|---|
| ScanNet | scene-level support and reported-score diagnostics | `SCANNET_ROOT` | official preprocessing may be required for the chosen backbone | See the official dataset terms; verify before camera-ready. |
| ARKitScenes | Concerto target-swap and coordinate-closure diagnostics | `ARKITSCENES_ROOT` | feature/cache extraction may be required | See the official dataset terms; verify before camera-ready. |
| ScanNet++ | Concerto target-swap and coordinate-closure diagnostics | `SCANNETPP_ROOT` | feature/cache extraction may be required | See the official dataset terms; verify before camera-ready. |
| S3DIS | Concerto target-swap and coordinate-closure diagnostics | `S3DIS_ROOT` | dataset-specific preprocessing may be required | See the official dataset terms; verify before camera-ready. |
| HM3D | Concerto target-swap and coordinate-closure diagnostics | `HM3D_ROOT` | feature/cache extraction may be required | See the official dataset terms; verify before camera-ready. |
| Structured3D | Concerto target-swap and coordinate-closure diagnostics | `STRUCTURED3D_ROOT` | feature/cache extraction may be required | See the official dataset terms; verify before camera-ready. |
| ScanObjectNN | object classification, checkpoint reporting, and PointGPT mask/order diagnostics | `SCANOBJECTNN_ROOT` | use the official or method-compatible H5 layout | See the official dataset terms; verify before camera-ready. |
| ShapeNetPart | object part segmentation support diagnostics | `SHAPENETPART_ROOT` | use the official part annotation layout | See the official dataset terms; verify before camera-ready. |

## Expected Local Layouts

### ScanNet

```text
$SCANNET_ROOT/
  scans/
  meta_data/
  preprocessed/        # optional, depending on the external codebase
```

### ARKitScenes

```text
$ARKITSCENES_ROOT/
  raw_or_release_files/
  preprocessed/        # optional
  feature_cache/       # optional
```

### ScanNet++

```text
$SCANNETPP_ROOT/
  data/
  splits/
  preprocessed/        # optional
```

### S3DIS

```text
$S3DIS_ROOT/
  Area_1/
  Area_2/
  ...
  preprocessed/        # optional
```

### HM3D

```text
$HM3D_ROOT/
  scenes/
  splits/
  feature_cache/       # optional
```

### Structured3D

```text
$STRUCTURED3D_ROOT/
  raw_or_release_files/
  splits/
  feature_cache/       # optional
```

### ScanObjectNN

```text
$SCANOBJECTNN_ROOT/
  h5_files/
    main_split/
    main_split_nobg/
```

For PB_T50_RS, use the method-compatible hardest-split H5 files expected by
the external PointGPT/Point-MAE/PCP-MAE code.

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
preprocessing is required, run the official preprocessing pipeline for the
corresponding dataset and model before launching the full diagnostics.
