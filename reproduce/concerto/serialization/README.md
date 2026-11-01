# R5: controlled PTv3 serialization-locality ladder

Set these fields in the PTv3 `backbone` config. Concerto passes the same
backbone config into its student and teacher builders.

```python
backbone = dict(
    # existing PTv3 fields ...
    attention_order_mode=None,
    attention_order_seed=20260724,
    attention_order_diagnostics=True,
)
```

The registered evaluation conditions are:

| Condition | `attention_order_mode` | Geometric code / pooling |
|---|---|---|
| Released baseline | absent, `None`, or `"off"` | Original four families and original `shuffle_orders` behavior |
| Forced z-only at eval | `"z_only"` | Valid z code at every stage |
| Forced Hilbert-only at eval | `"hilbert_only"` | Valid Hilbert code at every stage |
| Locality-preserving control | `"locality_preserving"` | Four valid families; deterministic family permutation shared with the destructive arm |
| Locality-destroying endpoint | `"locality_destroying"` | Identical to the preserving arm |

The two locality arms must use the same `attention_order_seed`. Their geometric
family permutations are fixed per stage without consuming global RNG. Only the
padded attention layout differs:

- `locality_preserving` permutes token occurrences inside each fixed window;
- `locality_destroying` permutes padded token occurrences inside each batch
  element and tracks the unique canonical occurrence used by the inverse.

Neither arm writes to `serialized_code`, `serialized_order`, or
`serialized_inverse`. `SerializedPooling` therefore derives the same
right-shifted codes, parents, and clusters in the matched pair.

Each intervened backbone exposes the latest audit records at
`backbone.last_attention_order_diagnostics` and on the returned `Point` as
`attention_order_diagnostics`. There is one record per encountered
`(stage, order_index, patch_size)`. Reject a run if:

- the preserving record has `locality_verified=False`;
- a destructive record has `destruction_possible=True` but
  `locality_verified=False`;
- batch, inverse, or shape assertions raise;
- the preserving arm drops similarly to the destructive arm.

`destruction_possible=False` means that batch element has only one attention
window at that stage; that stage cannot supply evidence for locality
destruction.

The primary report must pair frozen-checkpoint Concerto pretext loss with the
ScanNet frozen-backbone linear-probe result. The load-bearing number is the
linear-probe mIoU contrast between the destructive and preserving arms. These
are evaluation-time sensitivity conditions; only a separately matched
continuation/training experiment can support a learning-time claim.

The recorded values are in `results/paper_reported_results/concerto_serialization.csv`.
