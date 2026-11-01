#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from main_variant_step05_utils import (
    build_loader,
    build_main_variant_model,
    iter_limited,
    move_batch_to_cuda,
    parse_csv,
    repo_root_from_here,
    save_json,
    seed_everything,
    select_dataset_specs,
)


GLOBAL_TARGET_PERMUTATION = "global_target_permutation"
DIFFERENT_IMAGE_TARGET_SWAP = "different_image_target_swap"
SAME_SCENE_DIFFERENT_IMAGE_TARGET_SWAP = (
    "same_scene_different_image_target_swap"
)
CROSS_SCENE_TARGET_SWAP = "cross_scene_target_swap"
MODE_ALIASES = {
    "cross_image_target_swap": DIFFERENT_IMAGE_TARGET_SWAP,
}
MODES = [
    GLOBAL_TARGET_PERMUTATION,
    DIFFERENT_IMAGE_TARGET_SWAP,
    CROSS_SCENE_TARGET_SWAP,
    SAME_SCENE_DIFFERENT_IMAGE_TARGET_SWAP,
]


def normalize_modes(modes: list[str]) -> list[str]:
    normalized = []
    for mode in modes:
        canonical = MODE_ALIASES.get(str(mode), str(mode))
        if canonical not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        if canonical in normalized:
            raise ValueError(
                f"duplicate mode after alias normalization: {canonical}"
            )
        normalized.append(canonical)
    if not normalized:
        raise ValueError("at least one target-corruption mode is required")
    return normalized


def same_scene_different_image_target_swap(
    model,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    image_index: torch.Tensor,
) -> dict:
    """Derange image groups independently inside each eligible scene.

    A scene with fewer than two represented image groups is left byte-identical
    and marked skipped. No global, cross-scene, or same-image fallback is
    permitted. Eligible rows are returned explicitly so skipped rows cannot
    silently enter the distance aggregate as identity pairs.
    """
    if target.ndim < 1:
        raise ValueError("target must have a row dimension")
    if batch_index.shape != (target.shape[0],):
        raise ValueError("batch_index must have one value per target row")
    if image_index.shape != (target.shape[0],):
        raise ValueError("image_index must have one value per target row")
    if batch_index.device != target.device or image_index.device != target.device:
        raise ValueError("target and group indices must be on the same device")

    swapped = target.clone()
    eligible_row_mask = torch.zeros(
        target.shape[0], dtype=torch.bool, device=target.device
    )
    eligible_scenes = 0
    skipped_scenes = 0
    for scene in torch.unique(batch_index, sorted=True):
        scene_mask = batch_index == scene
        scene_images = torch.unique(image_index[scene_mask], sorted=True)
        if scene_images.numel() < 2:
            skipped_scenes += 1
            continue
        eligible_scenes += 1
        eligible_row_mask[scene_mask] = True
        permutation = model._sample_derangement(
            scene_images.numel(), target.device
        )
        for destination_position, source_position in enumerate(
            permutation.tolist()
        ):
            destination_image = scene_images[destination_position]
            source_image = scene_images[source_position]
            destination_mask = scene_mask & (
                image_index == destination_image
            )
            source_rows = target[
                scene_mask & (image_index == source_image)
            ]
            destination_count = int(destination_mask.sum().item())
            swapped[destination_mask] = (
                model._resample_rows_with_replacement(
                    source_rows,
                    destination_count,
                )
            )
    if eligible_scenes + skipped_scenes != int(
        torch.unique(batch_index).numel()
    ):
        raise AssertionError("scene eligibility accounting is incomplete")
    return {
        "target": swapped,
        "eligible_row_mask": eligible_row_mask,
        "eligible_scenes": eligible_scenes,
        "skipped_scenes": skipped_scenes,
    }


def corrupt_target(
    model,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    image_index: torch.Tensor,
    mode: str,
):
    mode = MODE_ALIASES.get(str(mode), str(mode))
    if mode == GLOBAL_TARGET_PERMUTATION:
        return model._apply_global_target_permutation(
            target, mode_name="target_distance_global_target_permutation"
        )
    if mode == DIFFERENT_IMAGE_TARGET_SWAP:
        return model._apply_cross_image_target_swap(target, image_index)
    if mode == CROSS_SCENE_TARGET_SWAP:
        return model._apply_cross_scene_target_swap(target, batch_index)
    if mode == SAME_SCENE_DIFFERENT_IMAGE_TARGET_SWAP:
        return same_scene_different_image_target_swap(
            model,
            target,
            batch_index,
            image_index,
        )["target"]
    raise ValueError(f"unknown mode: {mode}")


def corruption_result(
    model,
    target: torch.Tensor,
    batch_index: torch.Tensor,
    image_index: torch.Tensor,
    mode: str,
) -> dict:
    mode = MODE_ALIASES.get(str(mode), str(mode))
    if mode == SAME_SCENE_DIFFERENT_IMAGE_TARGET_SWAP:
        return same_scene_different_image_target_swap(
            model,
            target,
            batch_index,
            image_index,
        )
    return {
        "target": corrupt_target(
            model,
            target,
            batch_index,
            image_index,
            mode,
        ),
        "eligible_row_mask": torch.ones(
            target.shape[0], dtype=torch.bool, device=target.device
        ),
        "eligible_scenes": None,
        "skipped_scenes": None,
    }


def group_index_for_mode(
    batch_index: torch.Tensor,
    image_index: torch.Tensor,
    mode: str,
):
    mode = MODE_ALIASES.get(str(mode), str(mode))
    if mode == DIFFERENT_IMAGE_TARGET_SWAP:
        return image_index
    if mode == CROSS_SCENE_TARGET_SWAP:
        return batch_index
    return None


def init_stats():
    return {
        "batches": 0,
        "rows": 0,
        "sum_cos": 0.0,
        "sum_cos2": 0.0,
        "fallback_batches": 0,
        "sum_unique_groups": 0.0,
        "eligible_scenes": 0,
        "skipped_scenes": 0,
        "eligible_batches": 0,
        "skipped_batches": 0,
    }


def update_stats(
    stats: dict,
    cos: torch.Tensor,
    unique_groups: int | None,
    eligible_scenes: int | None = None,
    skipped_scenes: int | None = None,
) -> None:
    cos_cpu = cos.detach().float().cpu()
    rows = int(cos_cpu.numel())
    stats["batches"] += 1
    stats["rows"] += rows
    stats["sum_cos"] += float(cos_cpu.sum().item())
    stats["sum_cos2"] += float(cos_cpu.square().sum().item())
    if unique_groups is not None:
        stats["sum_unique_groups"] += float(unique_groups)
        if unique_groups <= 1:
            stats["fallback_batches"] += 1
    if eligible_scenes is not None:
        stats["eligible_scenes"] += int(eligible_scenes)
        if int(eligible_scenes) > 0:
            stats["eligible_batches"] += 1
    if skipped_scenes is not None:
        stats["skipped_scenes"] += int(skipped_scenes)
        if int(skipped_scenes) > 0:
            stats["skipped_batches"] += 1


def finalize_stats(dataset: str, mode: str, stats: dict) -> dict:
    rows = int(stats["rows"])
    batches = int(stats["batches"])
    mean = stats["sum_cos"] / rows if rows else float("nan")
    variance = stats["sum_cos2"] / rows - mean * mean if rows else float("nan")
    std = math.sqrt(max(variance, 0.0)) if rows else float("nan")
    mean_unique_groups = (
        stats["sum_unique_groups"] / batches
        if batches
        and mode in (
            DIFFERENT_IMAGE_TARGET_SWAP,
            CROSS_SCENE_TARGET_SWAP,
        )
        else ""
    )
    scene_constrained = mode == SAME_SCENE_DIFFERENT_IMAGE_TARGET_SWAP
    return {
        "dataset": dataset,
        "mode": mode,
        "batches": batches,
        "rows": rows,
        "mean_cos_original_corrupted": mean,
        "std_cos_original_corrupted": std,
        "mean_one_minus_cos": 1.0 - mean if rows else float("nan"),
        "fallback_batches": int(stats["fallback_batches"]),
        "mean_unique_groups": mean_unique_groups,
        "eligible_scenes": (
            int(stats["eligible_scenes"]) if scene_constrained else ""
        ),
        "skipped_scenes": (
            int(stats["skipped_scenes"]) if scene_constrained else ""
        ),
        "eligible_batches": (
            int(stats["eligible_batches"]) if scene_constrained else ""
        ),
        "skipped_batches": (
            int(stats["skipped_batches"]) if scene_constrained else ""
        ),
    }


def format_float(value) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


def write_results(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "mode",
        "batches",
        "rows",
        "mean_cos_original_corrupted",
        "std_cos_original_corrupted",
        "mean_one_minus_cos",
        "fallback_batches",
        "mean_unique_groups",
        "eligible_scenes",
        "skipped_scenes",
        "eligible_batches",
        "skipped_batches",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "mode": row["mode"],
                    "batches": row["batches"],
                    "rows": row["rows"],
                    "mean_cos_original_corrupted": format_float(
                        row["mean_cos_original_corrupted"]
                    ),
                    "std_cos_original_corrupted": format_float(
                        row["std_cos_original_corrupted"]
                    ),
                    "mean_one_minus_cos": format_float(row["mean_one_minus_cos"]),
                    "fallback_batches": row["fallback_batches"],
                    "mean_unique_groups": format_float(row["mean_unique_groups"]),
                    "eligible_scenes": row["eligible_scenes"],
                    "skipped_scenes": row["skipped_scenes"],
                    "eligible_batches": row["eligible_batches"],
                    "skipped_batches": row["skipped_batches"],
                }
            )

    lines = [
        "# Target Corruption Distance",
        "",
        "`mean cos` is `cos(t_original, t_corrupted)` after the same corruption helper used by the causal battery. Higher values mean the corrupted target stayed closer to the original target.",
        "",
        "| dataset | mode | batches | rows | mean cos | 1 - mean cos | std cos | fallback batches | mean unique groups | eligible scenes | skipped scenes | eligible batches | skipped batches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['mode']} | {row['batches']} | {row['rows']} | "
            f"{format_float(row['mean_cos_original_corrupted'])} | "
            f"{format_float(row['mean_one_minus_cos'])} | "
            f"{format_float(row['std_cos_original_corrupted'])} | "
            f"{row['fallback_batches']} | {format_float(row['mean_unique_groups'])} | "
            f"{row['eligible_scenes']} | {row['skipped_scenes']} | "
            f"{row['eligible_batches']} | {row['skipped_batches']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure cos(t_original, t_corrupted) for target-swap modes."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_here())
    parser.add_argument("--config", default="pretrain-concerto-v1m1-0-probe-enc2d-baseline")
    parser.add_argument("--weight", type=Path, default=None)
    parser.add_argument("--datasets", default="arkit,scannet")
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=(
            "Comma list of target-corruption modes. "
            "cross_image_target_swap is accepted as an input alias for "
            "different_image_target_swap."
        ),
    )
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default="main-origin-six-step05")
    parser.add_argument("--max-batches-per-dataset", type=int, default=32)
    parser.add_argument("--max-rows-per-batch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    modes = normalize_modes(parse_csv(args.modes))
    seed_everything(args.seed)
    if args.weight is None:
        args.weight = repo_root / "data" / "weights" / "concerto" / "concerto_base_origin.pth"
    if args.output_root is None:
        args.output_root = (
            repo_root
            / "data"
            / "runs"
            / "main_variant_target_corruption_distance"
            / args.tag
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg, model, load_info = build_main_variant_model(repo_root, args.config, args.weight)
    model.eval()
    specs = select_dataset_specs(
        repo_root,
        parse_csv(args.datasets),
        allow_missing=args.allow_missing_datasets,
    )
    loaders = {
        spec.name: build_loader(
            cfg,
            spec,
            "val",
            batch_size=args.batch_size,
            num_worker=args.num_worker,
            shuffle=False,
        )
        for spec in specs
    }

    aggregate = defaultdict(init_stats)
    with torch.inference_mode():
        for dataset_name, loader in loaders.items():
            for batch_idx, batch in iter_limited(loader, args.max_batches_per_dataset):
                batch = move_batch_to_cuda(batch)
                extracted = model.extract_enc2d_coord_target(
                    batch, max_rows=args.max_rows_per_batch
                )
                target = extracted["target"].detach().float()
                if target.numel() == 0:
                    continue
                batch_index = extracted["batch_index"].detach().long()
                image_index = extracted["image_index"].detach().long()
                for mode in modes:
                    group_index = group_index_for_mode(batch_index, image_index, mode)
                    unique_groups = (
                        int(torch.unique(group_index).numel())
                        if group_index is not None
                        else None
                    )
                    result = corruption_result(
                        model,
                        target,
                        batch_index,
                        image_index,
                        mode,
                    )
                    eligible = result["eligible_row_mask"]
                    corrupted = result["target"]
                    cos = F.cosine_similarity(
                        target[eligible],
                        corrupted[eligible].float(),
                        dim=1,
                        eps=1e-6,
                    )
                    update_stats(
                        aggregate[(dataset_name, mode)],
                        cos,
                        unique_groups,
                        eligible_scenes=result["eligible_scenes"],
                        skipped_scenes=result["skipped_scenes"],
                    )
                if (batch_idx + 1) % 8 == 0:
                    print(f"[distance] dataset={dataset_name} batches={batch_idx + 1}", flush=True)

    rows = [
        finalize_stats(dataset, mode, aggregate[(dataset, mode)])
        for dataset in [spec.name for spec in specs]
        for mode in modes
    ]
    csv_path = args.output_root / "results_target_corruption_distance.csv"
    md_path = args.output_root / "results_target_corruption_distance.md"
    write_results(rows, csv_path, md_path)
    save_json(
        args.output_root / "metadata.json",
        {
            "kind": "target_corruption_distance",
            "config": args.config,
            "source_weight": str(args.weight.resolve()),
            "load_info": load_info,
            "datasets": [spec.name for spec in specs],
            "modes": modes,
            "mode_aliases": MODE_ALIASES,
            "max_batches_per_dataset": args.max_batches_per_dataset,
            "max_rows_per_batch": args.max_rows_per_batch,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "csv": str(csv_path),
            "md": str(md_path),
        },
    )
    print(f"[distance] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
