from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dapc_scatter(
    dapc_csv: Path,
    out_png: Path,
    x_axis: str = "LD1",
    y_axis: str = "LD2",
    color_by: str = "pop",
) -> None:
    """DAPC scatter plot using discriminant coordinates."""
    df = pd.read_csv(dapc_csv)
    if x_axis not in df.columns or y_axis not in df.columns:
        raise ValueError(f"DAPC CSV must have columns '{x_axis}' and '{y_axis}'.")

    x = df[x_axis].to_numpy()
    y = df[y_axis].to_numpy()
    labels = df[color_by].astype(str).to_numpy() if color_by in df.columns else None

    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(x, y, s=20, alpha=0.8)
    else:
        uniq = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        for i, lab in enumerate(uniq):
            mask = labels == lab
            plt.scatter(
                x[mask],
                y[mask],
                s=20,
                alpha=0.8,
                label=str(lab),
                color=cmap(i),
            )
        plt.legend(title=color_by, fontsize="small")

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title("DAPC scatter")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_fst_manhattan(
    fst_csv: Path,
    out_png: Path,
    value_col: str = "Fst",
    chrom_col: str = "chrom",
    pos_col: str = "pos",
) -> None:
    """Simple Manhattan-style plot for Fst."""
    df = pd.read_csv(fst_csv)
    if value_col not in df.columns:
        raise ValueError(f"Fst CSV must have column '{value_col}'.")

    df = df.copy()
    df[chrom_col] = df[chrom_col].astype(str)
    df = df.sort_values([chrom_col, pos_col])

    # Build cumulative positions per chromosome.
    chroms = df[chrom_col].unique()
    offsets = {}
    current = 0
    for c in chroms:
        sub = df[df[chrom_col] == c]
        offsets[c] = current
        current += sub[pos_col].max() - sub[pos_col].min() + 1

    x = df[pos_col].to_numpy() + np.array([offsets[c] for c in df[chrom_col]])
    y = df[value_col].to_numpy()

    plt.figure(figsize=(10, 4))
    colors = plt.cm.get_cmap("tab20", len(chroms))
    for i, c in enumerate(chroms):
        mask = df[chrom_col] == c
        plt.scatter(
            x[mask],
            y[mask],
            s=6,
            color=colors(i),
            label=str(c),
            alpha=0.8,
        )

    plt.xlabel("Genomic position")
    plt.ylabel(value_col)
    plt.title("Fst Manhattan plot")
    plt.legend(title="chrom", fontsize="x-small", ncol=min(len(chroms), 6))
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_hw_dis_maf(
    hw_csv: Path,
    out_png: Path,
    pop: Optional[str] = None,
    maf_col: str = "maf",
    hw_col: str = "HWdis",
    pop_col: str = "population",
) -> None:
    """Scatter of HW disequilibrium vs MAF, optionally for a single population."""
    df = pd.read_csv(hw_csv)
    if maf_col not in df.columns or hw_col not in df.columns:
        raise ValueError(f"HW CSV must have '{maf_col}' and '{hw_col}' columns.")

    if pop is not None and pop_col in df.columns:
        df = df[df[pop_col] == pop]

    x = df[maf_col].to_numpy()
    y = df[hw_col].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=8, alpha=0.8)
    plt.xlabel("Minor allele frequency")
    plt.ylabel("Hardy–Weinberg disequilibrium")
    if pop is not None:
        plt.title(f"HW disequilibrium vs MAF (pop={pop})")
    else:
        plt.title("HW disequilibrium vs MAF")
    plt.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

