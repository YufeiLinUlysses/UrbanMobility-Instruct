"""Generate the revision-v5 dataset-composition figure used in the manuscript."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PRIMARY_SOURCES = {
    "highD_instructions": "highD",
    "ngsim_instructions": "NGSIM",
    "road_networks_instructions": "Road\nnetworks",
    "tlc_trip_instructions": "NYC TLC",
    "urban_flow_prediction_survey_instructions": "Urban-flow\nsurvey",
}
LEVEL_COLORS = {1: "#4472C4", 2: "#70AD47", 3: "#ED7D31"}
FORMAT_COLORS = {"Multiple choice": "#5B9BD5", "Short answer": "#A5A5A5"}
OBJECTIVE_COLORS = {
    "Answer\ngeneration": "#4472C4",
    "Candidate\nverification": "#70AD47",
    "Structured\nparaphrase": "#ED7D31",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _primary_counts(
    assembly: dict[str, Any],
) -> tuple[dict[str, Counter[int]], Counter[str], dict[str, Counter[int]]]:
    """Aggregate primary canonical records by source, level, and answer format."""

    by_source: dict[str, Counter[int]] = defaultdict(Counter)
    by_format: Counter[str] = Counter()
    by_format_level: dict[str, Counter[int]] = defaultdict(Counter)
    for key, count in assembly["counts"].items():
        source, task_type, level, _origin = key.split(":")
        if source not in PRIMARY_SOURCES:
            continue
        level_number = int(level.removeprefix("L"))
        by_source[source][level_number] += int(count)
        label = "Multiple choice" if task_type == "multiple_choice" else "Short answer"
        by_format[label] += int(count)
        by_format_level[label][level_number] += int(count)
    return by_source, by_format, by_format_level


def _training_objective_counts(split_manifest: dict[str, Any]) -> dict[str, int]:
    """Return the number of serialized training examples for each learning objective."""

    generation = int(split_manifest["generation_record_counts"]["train"])
    paraphrase = int(split_manifest["paraphrase_record_counts"]["train"])
    return {
        "Answer\ngeneration": generation,
        "Candidate\nverification": 2 * generation,
        "Structured\nparaphrase": paraphrase,
    }


def plot_dataset_composition(
    assembly_report_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot primary-source composition and the serialized training-objective mix.

    The left and middle panels describe the 10,560 canonical answer-generation records
    from the five primary sources. The right panel describes the training examples after
    those records are materialized into generation, verification, and paraphrase views.
    These views are related derivatives of canonical records, not independent concepts.
    """

    assembly = _read_json(Path(assembly_report_path))
    split_manifest = _read_json(Path(split_manifest_path))
    by_source, by_format, by_format_level = _primary_counts(assembly)
    objective_counts = _training_objective_counts(split_manifest)

    primary_total = sum(sum(counts.values()) for counts in by_source.values())
    if primary_total != 10_560:
        raise ValueError(f"Expected 10,560 primary records, found {primary_total:,}")
    if sum(by_format.values()) != primary_total:
        raise ValueError("Answer-format counts do not match the primary total")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.1),
        gridspec_kw={"width_ratios": [1.35, 1.35, 1.0]},
    )

    # (a) Canonical records by primary source and reasoning level.
    source_ids = list(PRIMARY_SOURCES)
    x_positions = range(len(source_ids))
    bottoms = [0] * len(source_ids)
    for level in (1, 2, 3):
        values = [by_source[source][level] for source in source_ids]
        axes[0].bar(
            x_positions,
            values,
            bottom=bottoms,
            color=LEVEL_COLORS[level],
            label=f"Level {level}",
            width=0.72,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    for index, total in enumerate(bottoms):
        axes[0].text(index, total + 42, f"{total:,}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(list(x_positions), [PRIMARY_SOURCES[source] for source in source_ids])
    axes[0].set_ylabel("Canonical records")
    axes[0].set_ylim(0, max(bottoms) * 1.15)
    axes[0].set_title("(a) Primary sources and reasoning levels")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    axes[0].grid(axis="y", alpha=0.2)

    # (b) Nested composition: answer format inside, reasoning level outside.
    format_labels = ["Multiple choice", "Short answer"]
    inner_values = [by_format[label] for label in format_labels]
    inner_wedges, _, _ = axes[1].pie(
        inner_values,
        radius=0.72,
        colors=[FORMAT_COLORS[label] for label in format_labels],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.70,
        textprops={"fontsize": 7.5, "fontweight": "bold"},
    )
    outer_values: list[int] = []
    outer_labels: list[str] = []
    outer_colors: list[str] = []
    for format_label in format_labels:
        for level in (1, 2, 3):
            value = by_format_level[format_label][level]
            if value:
                outer_values.append(value)
                outer_labels.append(f"L{level}\n{100 * value / primary_total:.1f}%")
                outer_colors.append(LEVEL_COLORS[level])
    axes[1].pie(
        outer_values,
        radius=1.0,
        labels=outer_labels,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        labeldistance=0.76,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 7.0},
    )
    axes[1].set_title("(b) Formats and reasoning levels")
    axes[1].legend(
        inner_wedges,
        ["MC", "SA"],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
    )

    # (c) Serialized examples actually supplied to training.
    objective_labels = list(objective_counts)
    objective_values = [objective_counts[label] for label in objective_labels]
    bars = axes[2].bar(
        objective_labels,
        objective_values,
        color=[OBJECTIVE_COLORS[label] for label in objective_labels],
        width=0.64,
    )
    axes[2].bar_label(bars, labels=[f"{value:,}" for value in objective_values], padding=3)
    axes[2].set_ylabel("Training examples")
    axes[2].set_ylim(0, max(objective_values) * 1.15)
    axes[2].set_title("(c) Training supervision views")
    axes[2].grid(axis="y", alpha=0.2)

    thousands = FuncFormatter(lambda value, _position: f"{value / 1000:g}k" if value else "0")
    for axis in (axes[0], axes[2]):
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_formatter(thousands)

    fig.tight_layout(w_pad=2.0)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def plot_separate_composition_panels(
    assembly_report_path: str | Path,
    split_manifest_path: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    """Export the three composition panels as independent PNG and PDF figures."""

    assembly = _read_json(Path(assembly_report_path))
    split_manifest = _read_json(Path(split_manifest_path))
    by_source, by_format, by_format_level = _primary_counts(assembly)
    objective_counts = _training_objective_counts(split_manifest)
    primary_total = sum(sum(counts.values()) for counts in by_source.values())
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    thousands = FuncFormatter(lambda value, _position: f"{value / 1000:g}k" if value else "0")

    # Standalone source-by-level stacked bars.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    source_ids = list(PRIMARY_SOURCES)
    x_positions = range(len(source_ids))
    bottoms = [0] * len(source_ids)
    for level in (1, 2, 3):
        values = [by_source[source][level] for source in source_ids]
        ax.bar(
            x_positions, values, bottom=bottoms, color=LEVEL_COLORS[level],
            label=f"Level {level}", width=0.72,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    for index, total in enumerate(bottoms):
        ax.text(index, total + 42, f"{total:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(x_positions), [PRIMARY_SOURCES[source] for source in source_ids])
    ax.set_ylabel("Canonical records")
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.set_title("Primary sources and reasoning levels")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(thousands)
    fig.tight_layout()
    source_path = destination / "dataset_composition_sources_levels.png"
    fig.savefig(source_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(source_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(source_path)

    # Standalone nested pie. Every label is placed inside its own ring.
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    format_labels = ["Multiple choice", "Short answer"]
    inner_values = [by_format[label] for label in format_labels]
    inner_labels = [
        f"MC\n{100 * inner_values[0] / primary_total:.1f}%",
        f"SA\n{100 * inner_values[1] / primary_total:.1f}%",
    ]
    ax.pie(
        inner_values,
        radius=0.72,
        labels=inner_labels,
        labeldistance=0.50,
        colors=[FORMAT_COLORS[label] for label in format_labels],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11, "fontweight": "bold", "ha": "center"},
    )
    outer_values: list[int] = []
    outer_labels: list[str] = []
    outer_colors: list[str] = []
    for format_label in format_labels:
        for level in (1, 2, 3):
            value = by_format_level[format_label][level]
            if value:
                outer_values.append(value)
                outer_labels.append(f"L{level}\n{100 * value / primary_total:.1f}%")
                outer_colors.append(LEVEL_COLORS[level])
    ax.pie(
        outer_values,
        radius=1.0,
        labels=outer_labels,
        labeldistance=0.84,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10, "ha": "center"},
    )
    ax.set_title("Answer formats and reasoning levels", pad=16)
    fig.tight_layout()
    pie_path = destination / "dataset_composition_formats_levels_pie.png"
    fig.savefig(pie_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pie_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(pie_path)

    # Standalone training-view bars.
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    objective_labels = list(objective_counts)
    objective_values = [objective_counts[label] for label in objective_labels]
    bars = ax.bar(
        objective_labels,
        objective_values,
        color=[OBJECTIVE_COLORS[label] for label in objective_labels],
        width=0.64,
    )
    ax.bar_label(bars, labels=[f"{value:,}" for value in objective_values], padding=3)
    ax.set_ylabel("Training examples")
    ax.set_ylim(0, max(objective_values) * 1.15)
    ax.set_title("Training supervision views")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(thousands)
    fig.tight_layout()
    views_path = destination / "dataset_composition_training_views.png"
    fig.savefig(views_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(views_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(views_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    reporting = repository_root / "urban_mobility" / "data" / "08_reporting"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-report",
        type=Path,
        default=reporting / "assembly_report.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=reporting / "experiment" / "split_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "submission" / "graphs" / "dataset_composition_revision_v5.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generated = plot_dataset_composition(
        arguments.assembly_report,
        arguments.split_manifest,
        arguments.output,
    )
    print(generated)
    for panel in plot_separate_composition_panels(
        arguments.assembly_report,
        arguments.split_manifest,
        arguments.output.parent,
    ):
        print(panel)
