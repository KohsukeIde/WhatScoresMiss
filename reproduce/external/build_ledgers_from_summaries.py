"""Create sanitized provenance summaries from compact paper-summary CSVs.

The generated CSVs are not raw model outputs, raw run logs, or checkpoint
artifacts. They preserve row-level provenance, configs, metrics, and source
scripts for review without committing raw datasets, large checkpoints, cluster
logs, or personal paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "paper_reported_results"
LEDGERS = ROOT / "results" / "ledgers"


BASE_COLUMNS = [
    "experiment_id",
    "model",
    "dataset",
    "split",
    "condition",
    "checkpoint_source",
    "seed",
    "metric",
    "value",
    "source_script",
    "config",
    "notes",
]


def write_ledger(name: str, rows: list[dict]) -> None:
    LEDGERS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[BASE_COLUMNS]
    df.to_csv(LEDGERS / name, index=False)


def concerto_loss_runs() -> None:
    src = pd.read_csv(SUMMARY / "concerto_pretext_diagnostics.csv")
    rows = []
    for _, r in src.iterrows():
        dataset = r["dataset"]
        config = f"reproduce/concerto/configs/{dataset.lower().replace('+', 'p')}_target_swap.yaml"
        deltas = {
            "clean": 0.0,
            "global_swap": r["global_swap_delta"],
            "cross_image_swap": r["cross_image_delta"],
            "cross_scene_swap": r["cross_scene_delta"],
        }
        for cond, delta in deltas.items():
            rows.append(
                {
                    "experiment_id": f"concerto_target_swap_{dataset}_{cond}",
                    "model": "Concerto-D",
                    "dataset": dataset,
                    "split": "paper_eval",
                    "condition": cond,
                    "checkpoint_source": "CONCERTO_CKPT",
                    "seed": "",
                    "metric": "pretext_loss",
                    "value": float(r["clean_loss"]) + float(delta),
                    "source_script": "reproduce/concerto/target_swap.py",
                    "config": config,
                    "notes": "paper-reported loss provenance summary; swap losses reconstructed from clean loss plus reported delta",
                }
            )
    write_ledger("concerto_loss_runs.csv", rows)


def concerto_closure_runs() -> None:
    src = pd.read_csv(SUMMARY / "concerto_pretext_diagnostics.csv")
    rows = []
    for _, r in src.iterrows():
        dataset = r["dataset"]
        clean = float(r["clean_loss"])
        mean_delta = float(pd.Series([r["global_swap_delta"], r["cross_image_delta"], r["cross_scene_delta"]]).mean())
        mean_swap = clean + mean_delta
        closure = float(r["xyz_closure_percent"]) / 100.0
        coord = mean_swap - closure * (mean_swap - clean)
        for cond, val in [("clean", clean), ("mean_swap", mean_swap), ("coordinate_only", coord)]:
            rows.append(
                {
                    "experiment_id": f"concerto_coordinate_closure_{dataset}_{cond}",
                    "model": "Concerto-D",
                    "dataset": dataset,
                    "split": "paper_eval",
                    "condition": cond,
                    "checkpoint_source": "CONCERTO_CKPT",
                    "metric": "pretext_loss",
                    "value": val,
                    "source_script": "reproduce/concerto/coordinate_only_closure.py",
                    "config": "configs/concerto/coordinate_closure.yaml",
                    "notes": "coordinate-only closure provenance summary reconstructed from paper-reported closure percentage",
                }
            )
    write_ledger("concerto_closure_fit_runs.csv", rows)


def representation_runs() -> None:
    src = pd.read_csv(SUMMARY / "concerto_representation_companion.csv")
    rows = []
    for _, r in src.iterrows():
        for metric in ["miou", "macc", "allacc", "delta_vs_cross_modal", "gap_recovery_percent"]:
            if pd.isna(r[metric]):
                continue
            rows.append(
                {
                    "experiment_id": f"concerto_representation_{r['continuation_target']}_{metric}",
                    "model": "Concerto-D",
                    "dataset": "ScanNet20",
                    "split": "linear_proxy",
                    "condition": r["continuation_target"],
                    "checkpoint_source": "CONCERTO_CKPT",
                    "metric": metric,
                    "value": r[metric],
                    "source_script": "reproduce/concerto/representation_companion_config.md",
                    "config": "configs/concerto/representation_companion.yaml",
                    "notes": "paper-reported representation companion row",
                }
            )
    write_ledger("concerto_representation_companion_runs.csv", rows)


def pointgpt_runs() -> None:
    src = pd.read_csv(SUMMARY / "pointgpt_mask_order_relaxations.csv")
    pre_rows = []
    down_rows = []
    for _, r in src.iterrows():
        cond = f"mask_{r['mask']}_order_{r['order']}"
        if not pd.isna(r["pretext_loss_drop_percent"]):
            pre_rows.append(
                {
                    "experiment_id": f"pointgpt_pretext_{cond}",
                    "model": "PointGPT-S",
                    "dataset": "ShapeNet55",
                    "split": "pretrain",
                    "condition": cond,
                    "checkpoint_source": "relaxed_variant_checkpoint",
                    "seed": 0,
                    "metric": "pretext_loss_drop_percent",
                    "value": r["pretext_loss_drop_percent"],
                    "source_script": "reproduce/pointgpt/scripts/train_relaxed_variant.sh",
                    "config": "configs/pointgpt/mask_order_relaxations.yaml",
                    "notes": "paper-reported pretext endpoint",
                }
            )
        for dataset, metric in [
            ("ScanObjectNN_obj_bg", "obj_bg_top1"),
            ("ScanObjectNN_obj_only", "obj_only_top1"),
            ("ScanObjectNN_PB_T50_RS", "pb_t50_rs_top1"),
            ("ShapeNetPart", "shapenetpart_inst_miou"),
        ]:
            down_rows.append(
                {
                    "experiment_id": f"pointgpt_downstream_{cond}_{dataset}",
                    "model": "PointGPT-S",
                    "dataset": dataset,
                    "split": "test",
                    "condition": cond,
                    "checkpoint_source": "relaxed_variant_ckpt_best",
                    "seed": 0,
                    "metric": "top1_percent" if dataset.startswith("ScanObjectNN") else "instance_miou_percent",
                    "value": r[metric],
                    "source_script": "reproduce/pointgpt/aggregate_pointgpt_results.py",
                    "config": "configs/pointgpt/mask_order_relaxations.yaml",
                    "notes": "paper-reported downstream evaluation",
                }
            )
    write_ledger("pointgpt_pretext_logs.csv", pre_rows)
    write_ledger("pointgpt_downstream_eval_runs.csv", down_rows)


def support_runs() -> None:
    scene = pd.read_csv(SUMMARY / "support_scannet_curves.csv")
    scene_rows = []
    for _, r in scene.iterrows():
        scene_rows.append(
            {
                "experiment_id": f"support_scannet_{r['model']}_{r['condition']}",
                "model": r["model"],
                "dataset": r["task"],
                "split": "validation",
                "condition": r["condition"],
                "checkpoint_source": "scene_ckpt_best",
                "metric": "miou_percent",
                "value": r["miou_percent"],
                "source_script": "reproduce/support/aggregate_support_results.py",
                "config": "configs/scene_support/scannet_support.yaml",
                "notes": f"score_space={r['score_space']}; source={r['source_result']}",
            }
        )
    write_ledger("scannet_support_eval_runs.csv", scene_rows)

    obj = pd.read_csv(SUMMARY / "support_shapenetpart_curves.csv")
    obj_rows = []
    for _, r in obj.iterrows():
        obj_rows.append(
            {
                "experiment_id": f"support_shapenetpart_{r['model']}_{r['condition']}",
                "model": r["model"],
                "dataset": r["task"],
                "split": "test",
                "condition": r["condition"],
                "checkpoint_source": r["selection_protocol"],
                "metric": "instance_miou_percent",
                "value": r["instance_miou_percent"],
                "seed": 0 if r["model"] == "PointGPT-S official" else "",
                "source_script": (
                    "PointGPT/segmentation/eval_shapenetpart_r7c_correction.py"
                    if r["model"] == "PointGPT-S official" else "paper-reported summary"
                ),
                "config": (
                    "original_replacement; support_preserving; replicates=2"
                    if r["model"] == "PointGPT-S official"
                    else "configs/object_support/shapenetpart_support.yaml"
                ),
                "notes": f"scope={r['metric_scope']}; aggregation={r['logit_aggregation']}; source={r['source_result']}",
            }
        )
    write_ledger("shapenetpart_support_eval_runs.csv", obj_rows)


def reported_score_runs() -> None:
    oracle = pd.read_csv(SUMMARY / "oracle_scannet.csv")
    rows = []
    for _, r in oracle.iterrows():
        for metric in ["top1_miou", "oracle2_miou", "oracle5_miou", "best_posthoc_miou"]:
            rows.append(
                {
                    "experiment_id": f"oracle_scannet_{r['model']}_{metric}",
                    "model": r["model"],
                    "dataset": "ScanNet20",
                    "split": "validation",
                    "condition": "reported_score",
                    "checkpoint_source": "scene_ckpt_best",
                    "metric": metric,
                    "value": r[metric],
                    "source_script": "reproduce/reported_score/oracle_from_logits.py",
                    "config": "configs/reported_score/oracle_scannet.yaml",
                    "notes": "paper-reported Oracle@k/posthoc diagnostic",
                }
            )
    picture = pd.read_csv(SUMMARY / "picture_wall_oracle_scannet.csv")
    for _, r in picture.iterrows():
        for metric in ["top1_picture_iou", "oracle2_picture_iou", "oracle5_picture_iou"]:
            rows.append(
                {
                    "experiment_id": f"picture_wall_oracle_{r['model']}_{metric}",
                    "model": r["model"],
                    "dataset": "ScanNet20",
                    "split": "validation",
                    "condition": "picture_wall",
                    "checkpoint_source": "scene_ckpt_best",
                    "metric": metric,
                    "value": r[metric],
                    "source_script": "reproduce/reported_score/oracle_from_logits.py",
                    "config": "configs/reported_score/oracle_scannet.yaml",
                    "notes": "paper-reported picture/wall Oracle@k diagnostic",
                }
            )
    write_ledger("scannet_oracle_eval_runs.csv", rows)

    class_pair = pd.read_csv(SUMMARY / "class_pair_diagnostics_scannet.csv")
    cp_rows = []
    for _, r in class_pair.iterrows():
        for metric in ["fp_ba", "head_ba", "target_to_confusion"]:
            cp_rows.append(
                {
                    "experiment_id": f"class_pair_{r['pair']}_{r['model']}_{metric}",
                    "model": r["model"],
                    "dataset": "ScanNet20",
                    "split": "validation",
                    "condition": r["pair"],
                    "checkpoint_source": "scene_ckpt_best",
                    "metric": metric,
                    "value": r[metric],
                    "source_script": "reproduce/reported_score/class_pair_diagnostics.py",
                    "config": "configs/reported_score/class_pair_diagnostics.yaml",
                    "notes": "paper-reported class-pair diagnostic",
                }
            )
    write_ledger("class_pair_diagnostic_runs.csv", cp_rows)

    ckpt = pd.read_csv(SUMMARY / "checkpoint_reporting_scanobjectnn.csv")
    ckpt_rows = []
    for _, r in ckpt.iterrows():
        for split in ["obj_bg", "obj_only", "pb_t50_rs"]:
            ckpt_rows.append(
                {
                    "experiment_id": f"checkpoint_reporting_{r['model']}_{r['checkpoint_protocol']}_{split}",
                    "model": r["model"],
                    "dataset": f"ScanObjectNN_{split}",
                    "split": "test",
                    "condition": r["checkpoint_protocol"],
                    "checkpoint_source": r["checkpoint_protocol"],
                    "metric": "top1_percent",
                    "value": r[split],
                    "source_script": "reproduce/reported_score/checkpoint_reporting.py",
                    "config": "configs/reported_score/checkpoint_reporting.yaml",
                    "notes": "paper-reported checkpoint protocol diagnostic",
                }
            )
    write_ledger("scanobjectnn_checkpoint_selection_runs.csv", ckpt_rows)


def main() -> None:
    concerto_loss_runs()
    concerto_closure_runs()
    representation_runs()
    pointgpt_runs()
    support_runs()
    reported_score_runs()
    print(f"Wrote provenance summaries to {LEDGERS}")


if __name__ == "__main__":
    main()
