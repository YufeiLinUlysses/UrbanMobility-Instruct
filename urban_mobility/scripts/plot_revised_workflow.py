"""Render the revised dataset construction and evaluation workflow for Figure 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "submission" / "plots"

COLORS = {
    "red": "#E53935",
    "amber": "#F2A900",
    "green": "#00A651",
    "blue": "#2B5CAA",
    "purple": "#7B2CBF",
    "teal": "#008C95",
    "gray": "#58636F",
    "light": "#F7F8FA",
    "ink": "#16191D",
}


def rounded_box(ax, x, y, w, h, title, body, color, *, title_size=13, body_size=10.5,
                fill="white", linewidth=1.8, title_align="left"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        facecolor=fill,
        edgecolor=color,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(box)
    tx = x + (w / 2 if title_align == "center" else 0.018)
    ha = "center" if title_align == "center" else "left"
    ax.text(tx, y + h - 0.034, title, ha=ha, va="top", fontsize=title_size,
            fontweight="bold", color=color, zorder=4)
    ax.text(x + 0.018, y + h - 0.085, body, ha="left", va="top",
            fontsize=body_size, color=COLORS["ink"], linespacing=1.25, zorder=4)
    return box


def arrow(ax, start, end, *, color="#1F2328", width=1.8, style="-|>", dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=width, color=color,
        linestyle="--" if dashed else "-", zorder=2,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)


def elbow_arrow(ax, points, *, color="#1F2328", width=1.8):
    """Draw an orthogonal connector whose final segment carries the arrowhead."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                linewidth=width, zorder=2)
    arrow(ax, points[-2], points[-1], color=color, width=width)


def label(ax, x, y, text, color, *, size=11):
    ax.text(
        x, y, text, ha="center", va="center", fontsize=size,
        fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor=color),
        zorder=5,
    )


def render() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "data_construction_evaluation_workflow_revision_v5.png"
    pdf_path = OUTPUT_DIR / "data_construction_evaluation_workflow_revision_v5.pdf"

    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.955, "Dataset construction, model adaptation, and evaluation workflow",
            fontsize=24, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.035, 0.912,
            "Grounded seed families are split before augmentation; external sources enter only at evaluation.",
            fontsize=12.5, color=COLORS["gray"], va="top")

    label(ax, 0.065, 0.845, "CONSTRUCTION", COLORS["red"])
    label(ax, 0.395, 0.845, "LEAKAGE CONTROL", COLORS["amber"])
    label(ax, 0.675, 0.845, "TASK VIEWS", COLORS["purple"])
    label(ax, 0.905, 0.845, "EVALUATION", COLORS["blue"])

    rounded_box(
        ax, 0.035, 0.635, 0.18, 0.16,
        "1  Ground source information",
        "Five heterogeneous sources\n"
        "• trajectories and traffic\n"
        "• road-network structure\n"
        "• mobility demand and survey evidence",
        COLORS["red"], title_size=11.5, body_size=9.8,
    )
    rounded_box(
        ax, 0.245, 0.635, 0.18, 0.16,
        "2  Author seed instructions",
        "Fact • question • options\n"
        "answer • explanation • provenance\n"
        "Level 1–3 and analytical perspective",
        COLORS["amber"], title_size=11.5, body_size=9.8,
    )
    rounded_box(
        ax, 0.455, 0.635, 0.18, 0.16,
        "3  Assign stable identities",
        "seed_id identifies one authored item\n"
        "concept_group_id keeps related items\n"
        "together throughout the pipeline",
        COLORS["green"], title_size=11.5, body_size=9.8,
    )
    rounded_box(
        ax, 0.665, 0.635, 0.30, 0.16,
        "4  Split concepts before augmentation",
        "Training 80%        Validation 10%        In-domain test 10%\n"
        "All descendants inherit the split of their originating concept group.",
        COLORS["blue"], title_size=11.5, body_size=9.8, fill="#F5F8FD",
    )
    arrow(ax, (0.215, 0.715), (0.245, 0.715))
    arrow(ax, (0.425, 0.715), (0.455, 0.715))
    arrow(ax, (0.635, 0.715), (0.665, 0.715))

    ax.text(0.815, 0.605, "SPLIT BEFORE AUGMENTATION", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=COLORS["blue"])

    rounded_box(
        ax, 0.12, 0.395, 0.25, 0.145,
        "5  Generate controlled descendants",
        "Vary wording and option order while preserving\n"
        "the source claim, answer, level, task identity,\n"
        "seed identity, and concept membership.",
        COLORS["purple"], body_size=10.5,
    )
    rounded_box(
        ax, 0.405, 0.395, 0.21, 0.145,
        "6  Validate records",
        "Schema compliance • answer consistency\n"
        "field preservation • provenance\n"
        "semantic and leakage audits",
        COLORS["teal"], body_size=10.5,
    )
    arrow(ax, (0.815, 0.635), (0.37, 0.54), color=COLORS["blue"])
    arrow(ax, (0.37, 0.468), (0.405, 0.468))

    rounded_box(
        ax, 0.65, 0.405, 0.10, 0.12,
        "Generation",
        "Answer + grounded\nexplanation",
        COLORS["purple"], title_size=10.5, body_size=8.7,
    )
    rounded_box(
        ax, 0.77, 0.405, 0.10, 0.12,
        "Verification",
        "Supported vs.\nunsupported candidate",
        COLORS["purple"], title_size=10.5, body_size=8.4,
    )
    rounded_box(
        ax, 0.89, 0.405, 0.08, 0.12,
        "Paraphrase",
        "Preserve answer\nand structure",
        COLORS["purple"], title_size=9.8, body_size=8.2,
    )
    arrow(ax, (0.615, 0.468), (0.65, 0.468))
    arrow(ax, (0.632, 0.468), (0.77, 0.468))
    arrow(ax, (0.64, 0.468), (0.89, 0.468))

    rounded_box(
        ax, 0.035, 0.135, 0.19, 0.15,
        "Model adaptation",
        "Fine-tune on training views only\n"
        "T5-base • Qwen 7B/14B\n"
        "Llama 3.1 8B/70B",
        COLORS["green"], body_size=10.5, fill="#F4FBF7",
    )
    rounded_box(
        ax, 0.265, 0.135, 0.20, 0.15,
        "In-domain evaluation",
        "Held-out primary-source concepts\n"
        "Tests transfer beyond training seeds\n"
        "within the represented sources",
        COLORS["blue"], body_size=10.2, fill="#F5F8FD",
    )
    rounded_box(
        ax, 0.505, 0.135, 0.20, 0.15,
        "France + Japan holdout",
        "66 new seeds plus descendants\n"
        "Tests cross-regional transfer and\n"
        "robustness to question variants",
        COLORS["blue"], body_size=10.2, fill="#F5F8FD",
    )
    rounded_box(
        ax, 0.745, 0.135, 0.22, 0.15,
        "Ten-dataset multi-regional test",
        "Entirely new sources, facts, and seeds\n"
        "Tests broader geographic and\n"
        "dataset-level transfer",
        COLORS["blue"], title_size=11.5, body_size=10.0, fill="#F5F8FD",
    )
    elbow_arrow(ax, [(0.81, 0.405), (0.81, 0.34), (0.13, 0.34), (0.13, 0.285)],
                color=COLORS["purple"])
    arrow(ax, (0.225, 0.21), (0.265, 0.21))
    arrow(ax, (0.225, 0.21), (0.505, 0.21))
    arrow(ax, (0.225, 0.21), (0.745, 0.21))

    ax.text(0.605, 0.085,
            "France, Japan, and the ten multi-regional sources are excluded from model fitting and checkpoint selection.",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=COLORS["blue"])
    ax.text(0.605, 0.047,
            "Base and fine-tuned checkpoints are compared under the same evaluation protocol; results are aggregated by row and underlying seed/source.",
            ha="center", va="center", fontsize=10.5, color=COLORS["gray"])

    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12, dpi=240)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = render()
    print(png)
    print(pdf)
