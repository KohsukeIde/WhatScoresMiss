"""Compare released target swaps against the original Concerto implementation."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
from pathlib import Path

import torch

from reproduce.concerto.target_swap_ops import TargetSwapOperator


def load_reference(path: Path):
    source = path.read_text()
    tree = ast.parse(source)
    concerto = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Concerto")
    names = (
        "_sample_derangement", "_resample_rows_with_replacement", "_log_shortcut_probe_once",
        "_apply_global_target_permutation", "_apply_group_target_swap",
        "_apply_cross_scene_target_swap", "_apply_cross_image_target_swap",
    )
    methods = {node.name: node for node in concerto.body if isinstance(node, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    code = "import torch\nclass Reference:\n    def __init__(self):\n        self._shortcut_probe_logs = set()\n"
    for name in names:
        node = methods[name]
        if name in ("_sample_derangement", "_resample_rows_with_replacement"):
            code += "    @staticmethod\n"
        code += "".join(lines[node.lineno - 1:node.end_lineno]) + "\n"
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["Reference"]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Original Concerto model source file")
    args = parser.parse_args()
    reference = load_reference(Path(args.reference))
    operator = TargetSwapOperator()
    modes = ("global_target_permutation", "cross_image_target_swap", "cross_scene_target_swap")
    checked = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for n in (1, 2, 3, 8, 19, 37):
            features = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)
            for groups in (torch.zeros(n, dtype=torch.long), torch.arange(n) % 2, torch.arange(n) % 5):
                for mode in modes:
                    for seed in range(12):
                        torch.manual_seed(seed)
                        actual = operator.apply(features, mode, groups)
                        torch.manual_seed(seed)
                        if mode == "global_target_permutation":
                            expected = reference._apply_global_target_permutation(features)
                        elif mode == "cross_image_target_swap":
                            expected = reference._apply_cross_image_target_swap(features, groups)
                        else:
                            expected = reference._apply_cross_scene_target_swap(features, groups)
                        if not torch.equal(actual, expected):
                            raise AssertionError(f"target swap mismatch: n={n}, mode={mode}, seed={seed}")
                        checked += 1
    print(f"PASS: {checked} target swaps match the original Concerto implementation")


if __name__ == "__main__":
    main()
