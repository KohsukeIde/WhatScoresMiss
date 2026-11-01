# Object-level rotation sensitivity

Evaluation code behind the rotation table in Appendix C. The model
implementations, datasets, and checkpoints are not bundled.

- `tools/eval_scanobjectnn_rotation_sensitivity.py`: ScanObjectNN `obj_bg` with
  rotations about the $z$-axis and random SO(3) rotations.
- `rotation_v2/eval_shapenetpart_rotation.py`: ShapeNetPart with rotations
  $45^\circ,\dots,315^\circ$ about the $z$-axis and eight SO(3) rotations per
  shape, each read against its own clean control.
- `rebuttal_eval/rotations.py`: rotation matrices and the SO(3) sampler. The
  scripts also use `determinism.py` and `r7.py` from
  [`../original_r7c/rebuttal_eval/`](../original_r7c/rebuttal_eval/).

Copy these directories into a PointGPT checkout prepared as described in
[`../original_r7c/README.md`](../original_r7c/README.md). The recorded values are
in `results/paper_reported_results/object_rotation_sensitivity.csv`.
