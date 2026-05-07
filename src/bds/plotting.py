"""Matplotlib figure builders for paper-reported CSV summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, output_path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_concerto_pretext_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(df["dataset"], df["xyz_closure_percent"])
    ax.set_ylabel("XYZ closure (%)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, output_path)


def make_support_scannet_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = range(len(df))
    ax.plot(x, df["clean_miou"], marker="o", label="clean")
    ax.plot(x, df["random50_miou"], marker="o", label="random50")
    ax.plot(x, df["structured50_miou"], marker="o", label="structured50")
    ax.set_xticks(list(x), df["model"], rotation=20)
    ax.set_ylabel("mIoU (%)")
    ax.legend()
    _save(fig, output_path)


def make_support_shapenetpart_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = range(len(df))
    ax.plot(x, df["clean_inst_miou"], marker="o", label="clean")
    ax.plot(x, df["random50_inst_miou"], marker="o", label="random50")
    ax.plot(x, df["structured50_inst_miou"], marker="o", label="structured50")
    ax.plot(x, df["semantic50_inst_miou"], marker="o", label="semantic50")
    ax.set_xticks(list(x), df["model"], rotation=20)
    ax.set_ylabel("Instance mIoU (%)")
    ax.legend()
    _save(fig, output_path)


def make_support_scannet_curves_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, perturbation in zip(axes, ["random_keep", "structured_keep"]):
        sub = df[df["perturbation"] == perturbation]
        for model, group in sub.groupby("model", sort=False):
            group = group.sort_values("keep_fraction")
            ax.plot(group["keep_fraction"], group["miou_percent"], marker="o", label=model)
        ax.set_title(perturbation.replace("_", " "))
        ax.set_xlabel("Retained fraction")
        ax.set_xticks([0.1, 0.2, 0.5, 0.8])
    axes[0].set_ylabel("mIoU (%)")
    axes[-1].legend(fontsize="small")
    _save(fig, output_path)


def make_support_shapenetpart_curves_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    perturbations = ["random_keep", "structured_keep", "semantic_keep_per_part"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, perturbation in zip(axes, perturbations):
        sub = df[df["perturbation"] == perturbation]
        for model, group in sub.groupby("model", sort=False):
            group = group.sort_values("keep_fraction")
            ax.plot(group["keep_fraction"], group["instance_miou_percent"], marker="o", label=model)
        ax.set_title(perturbation.replace("_", " "))
        ax.set_xlabel("Retained fraction")
        ax.set_xticks([0.1, 0.2, 0.5, 0.8])
    axes[0].set_ylabel("Instance mIoU (%)")
    axes[-1].legend(fontsize="small")
    _save(fig, output_path)


def make_oracle_scannet_figure(csv_path, output_path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = range(len(df))
    ax.plot(x, df["top1_miou"], marker="o", label="top1")
    ax.plot(x, df["oracle2_miou"], marker="o", label="Oracle@2")
    ax.plot(x, df["oracle5_miou"], marker="o", label="Oracle@5")
    ax.plot(x, df["best_posthoc_miou"], marker="o", label="best post-hoc")
    ax.set_xticks(list(x), df["model"], rotation=20)
    ax.set_ylabel("mIoU (%)")
    ax.legend()
    _save(fig, output_path)
