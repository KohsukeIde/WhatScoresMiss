#!/usr/bin/env python3
"""Generate one immutable rotation_matrix_v2 bank from explicit example IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rotation_v2.frozen_bank import (  # noqa: E402
    generate_frozen_bank,
    read_example_id_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new rotation_matrix_v2 frozen bank. The output path "
            "must not already exist."
        )
    )
    parser.add_argument(
        "--example-ids",
        required=True,
        type=Path,
        help="JSON list or one-ID-per-line UTF-8 text file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory to create. Existing paths are rejected.",
    )
    parser.add_argument(
        "--global-seed",
        required=True,
        type=int,
        help="Unsigned 64-bit global protocol seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example_ids = read_example_id_list(args.example_ids)
    bank = generate_frozen_bank(
        args.output_dir,
        example_ids,
        global_seed=args.global_seed,
    )
    summary = {
        "output_dir": str(bank.directory),
        "num_examples": len(bank.example_ids),
        "num_conditions": len(bank.conditions),
        "manifest_sha256": bank.manifest["manifest_sha256"],
        "content_sha256": bank.manifest["content_sha256"],
        "generator_source_sha256": bank.manifest["generator_source_sha256"],
    }
    print(
        json.dumps(
            summary,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
