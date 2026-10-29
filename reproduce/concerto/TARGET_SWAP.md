# Concerto target-swap intervention

The target-swap operator in `target_swap_ops.py` is extracted from the Concerto
implementation used in our rebuttal. It replaces 2D target features while
keeping the 3D predictions fixed. It implements the global, cross-image, and
cross-scene procedures described in Appendix B. When a batch contains only one
image or scene group, the group condition uses the global permutation.

The original implementation tries up to 32 random permutations before using a
cyclic derangement. This release raises an error if all 32 attempts fail rather
than silently changing the sampling rule. For all successful random draws,
the sampling procedure is unchanged.

`patches/concerto_target_swap.patch` applies to Pointcept commit
`1342eda30e96cbb5fadf5374ff8eb18f6de15c71`. Set `BDS_ROOT` to the absolute
path of this repository, then prepare a fresh Pointcept checkout:

```bash
export BDS_ROOT=/path/to/beyond-downstream-scores
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept
git checkout 1342eda30e96cbb5fadf5374ff8eb18f6de15c71
cp "$BDS_ROOT/reproduce/concerto/target_swap_ops.py" pointcept/models/concerto/bds_target_swap_ops.py
git apply "$BDS_ROOT/reproduce/concerto/patches/concerto_target_swap.patch"
```

Explicitly set `target_swap_mode` in the model config to one of `none`,
`global_target_permutation`, `cross_image_target_swap`, or
`cross_scene_target_swap`. The patch preserves the existing prediction path
and changes the target after feature grouping.

The condition names used in the result ledger map to model settings as follows:

| Ledger condition | `model.target_swap_mode` |
|---|---|
| `clean` | `none` |
| `global_swap` | `global_target_permutation` |
| `cross_image_swap` | `cross_image_target_swap` |
| `cross_scene_swap` | `cross_scene_target_swap` |

The patch provides the target-swap intervention, not a complete reproduction
of the six-dataset experiment. The short-refit 2D projection head, original
Pointcept data configuration, six datasets, and checkpoints must also be
prepared. `target_swap.py` aggregates the already-recorded loss ledger; it
does not launch model inference. Exact end-to-end reproduction of the reported
loss values has not been verified from this package.

For a synthetic parity check against the original model source, run:

```bash
python scripts/verify_concerto_swap.py --reference /path/to/concerto_v1m1_base.py
```

This checks the operator only; it does not run Concerto inference.
