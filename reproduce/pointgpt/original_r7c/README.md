# PointGPT ShapeNetPart retention replay

This directory contains the evaluation driver and helper modules used for the
PointGPT rebuttal retention replay. The model implementation, ShapeNetPart
point clouds, checkpoint, and CUDA extensions are not bundled. The released
`score_retained_points.py` implements the same per-shape scoring operation for
already-computed logits; it does not run PointGPT inference.

The helper modules are copied from the rebuttal run. The evaluation driver is
also from that run, with the four protocol arguments `--seed`, `--replicates`,
`--batch_size`, and `--base_sampling` made mandatory. This prevents an omitted
sampling option from selecting a different protocol. The grouping patch is
against PointGPT commit `9ba3d40e3aaa0933761639345b90db39c2295e96`; it contains
the grouping changes needed by the evaluator. Unlike the original cluster
copy, it does not use an import fallback for the Chamfer extension.

The following commands prepare a **fresh** PointGPT checkout. Set `BDS_ROOT`
to the absolute path of this repository before changing directories:

```bash
export BDS_ROOT=/path/to/beyond-downstream-scores
git clone https://github.com/CGuangyan-BIT/PointGPT.git
cd PointGPT
git checkout 9ba3d40e3aaa0933761639345b90db39c2295e96
git apply "$BDS_ROOT/reproduce/pointgpt/original_r7c/pointgpt_grouping.patch"
cp -R "$BDS_ROOT/reproduce/pointgpt/original_r7c/rebuttal_eval" .
cp "$BDS_ROOT/reproduce/pointgpt/original_r7c/segmentation/eval_shapenetpart_r7c_correction.py" segmentation/
```

Install PointGPT and its CUDA extensions according to the upstream project,
prepare the ShapeNetPart dataset and the PointGPT-S segmentation
checkpoint, then set `SHAPENETPART_ROOT`, `POINTGPT_CKPT`, and `OUTPUT_ROOT`.
The checkpoint used for the reported replay has SHA-256
`96079848c187d681f3a66aaf7860fa5a7b3a3882df66a4efddf5a9a5fd9c4002`.
It is not included in this repository. The checkpoint, input dataset, library
versions, and execution environment must match the evaluation setup to
reproduce its numbers. The following is the
rebuttal protocol, not a claim of bit-for-bit equality on other hardware:

```bash
mkdir -p "$OUTPUT_ROOT"
python segmentation/eval_shapenetpart_r7c_correction.py \\
  --ckpt "$POINTGPT_CKPT" \\
  --root "$SHAPENETPART_ROOT" \\
  --plan_dir "$OUTPUT_ROOT/frozen_plan" \\
  --output_json "$OUTPUT_ROOT/pointgpt_r7c.json" \\
  --trace_jsonl_gz "$OUTPUT_ROOT/pointgpt_r7c_trace.jsonl.gz" \\
  --long_csv "$OUTPUT_ROOT/pointgpt_r7c_curves.csv" \\
  --wide_csv "$OUTPUT_ROOT/pointgpt_r7c_summary.csv" \\
  --seed 0 --replicates 2 --batch_size 16 \\
  --base_sampling original_replacement
```

The model and clean input remain fixed. Two perturbation replicates are used
for each non-clean condition. Scores average logits for repeated raw point IDs,
restrict predictions to the shape category, calculate per-shape part mIoU, and
average over shapes. The actual evaluation script requires a GPU and the
original external dependencies; the lightweight parity check in the main
repository uses synthetic arrays and does not run the model.
