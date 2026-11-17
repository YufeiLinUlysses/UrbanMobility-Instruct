"""Refresh five-model prompt-cleaned manuscript evidence and loss figure."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "submission"
OUT = ROOT / "data/08_reporting/manuscript_refresh_2026-09-17"
MODEL_REPO = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
PRED_REPO = "UlyssesLynne/urban-planning-llm-predictions-v3"
RUNS = {
    "t5_base": ("revision_v6_prompt_cleaned", "revision-v6-t5-prompt-cleaned-v1"),
    "qwen25_7b": ("revision_v6_prompt_cleaned", "revision-v6-qwen25-7b-prompt-cleaned-v1"),
    "qwen25_14b": ("revision_v6_prompt_cleaned", "revision-v6-qwen25-14b-prompt-cleaned-v1"),
    "llama31_8b": ("revision_v6_prompt_cleaned", "revision-v6-llama31-8b-prompt-cleaned-v1"),
    "llama31_70b": ("revision_v6_prompt_cleaned", "revision-v6-llama31-70b-prompt-cleaned-v1"),
}
LABELS = {"t5_base": "T5-base", "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "llama31_8b": "Llama 3.1-8B", "llama31_70b": "Llama 3.1-70B"}


def download(repo: str, filename: str, repo_type: str, revision: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type=repo_type, revision=revision))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model_sha = api.repo_info(MODEL_REPO, repo_type="model").sha
    prediction_sha = api.repo_info(PRED_REPO, repo_type="dataset").sha
    frames = {}
    histories = {}
    manifests = {}
    for key, (version, run) in RUNS.items():
        path = f"evaluations/{version}/{key}/fine_tuned/in_domain/{run}-finetuned-in-domain/predictions.parquet"
        frame = pd.read_parquet(download(PRED_REPO, path, "dataset", prediction_sha))
        assert len(frame) == 5552 and frame.model_key.eq(key).all()
        frames[key] = frame
        if key in LABELS:
            prefix = f"{key}/{run}/checkpoint"
            histories[key] = json.loads(download(MODEL_REPO, prefix + "/training_history.json", "model", model_sha).read_text())
            manifests[key] = json.loads(download(MODEL_REPO, prefix + "/training_manifest.json", "model", model_sha).read_text())

    level = {}
    for key, frame in frames.items():
        level[key] = {}
        for task, column, scale in (("short_answer", "rougeL", 1),
                                    ("short_answer", "bertscore_f1", 1),
                                    ("short_answer", "nli_entailment_score", 1),
                                    ("short_answer", "exact_match", 100),
                                    ("short_answer_explanation", "bertscore_f1", 1),
                                    ("mc_answer", "normalized_correct", 100),
                                    ("answer_verification", "correct", 100),
                                    ("paraphrase", "answer_preserved", 100),
                                    ("paraphrase", "field_preservation", 100),
                                    ("paraphrase", "bertscore_f1", 1)):
            part = frame[frame.evaluation_task.eq(task)]
            level[key][f"{task}.{column}"] = {
                str(int(k)): {"count": len(g), "mean": float(g[column].mean() * scale)}
                for k, g in part.groupby("Level")
            }

    data = pd.concat(frames.values(), ignore_index=True)
    low = {}
    thresholds = {}
    for metric in ("rougeL", "bertscore_f1", "nli_entailment_score"):
        thresholds[metric] = {}
        for task, group in data.groupby("evaluation_task"):
            if metric == "rougeL" and task in ("mc_answer", "answer_verification"):
                continue
            scores = group[metric].dropna()
            cutoff = float(scores.quantile(.1, interpolation="linear"))
            thresholds[metric][task] = {
                "cutoff": cutoff, "count": len(scores),
                "below": int((scores < cutoff).sum()), "tied": int((scores == cutoff).sum()),
            }
        for key, frame in frames.items():
            below = denom = 0
            for task, cutoff_info in thresholds[metric].items():
                values = frame.loc[frame.evaluation_task.eq(task), metric].dropna()
                below += int((values < cutoff_info["cutoff"]).sum())
                denom += len(values)
            low.setdefault(key, {})[metric] = {
                "low_count": below, "denominator": denom, "percent": 100 * below / denom,
            }

    fig = plt.figure(figsize=(12.0, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5])]
    for ax, (key, label) in zip(axes, LABELS.items()):
        history = histories[key]
        train = [(row["step"], row["loss"]) for row in history if "loss" in row]
        validation_rows = [(row["step"], row["eval_loss"]) for row in history if "eval_loss" in row]
        # Trainer logs a final loss after restoring the best checkpoint at the
        # stopping step. Keep the first value at each step for the trajectory.
        validation = list(dict(reversed(validation_rows)).items())
        validation.sort()
        assert train and validation
        ax.plot(*zip(*train), color="#246596", linewidth=1.7, label="Training loss")
        ax.plot(*zip(*validation), color="#ce663f", marker="o", markersize=3.2,
                linewidth=1.4, label="Validation loss")
        best = min(validation, key=lambda pair: pair[1])
        ax.scatter(*best, color="#9d3d22", marker="*", s=90, zorder=4)
        ax.set_title(label)
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.2)
        ax.set_xlim(left=0)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(PAPER / f"plots/training_loss_prompt_cleaned.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "model_repo_commit": model_sha, "prediction_repo_commit": prediction_sha,
        "runs": RUNS, "training_manifests": manifests,
        "level_metrics": level, "low_score_rates": low, "low_score_thresholds": thresholds,
    }
    (OUT / "evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Loss figure:", PAPER / "plots/training_loss_prompt_cleaned.pdf")
    print("Training:", {key: {"loss": val["validation_metrics"]["eval_loss"],
                         "runtime": val["train_metrics"]["train_runtime"]} for key, val in manifests.items()})
    print("Low rates:", json.dumps(low, indent=2))
    print("Level metrics:", json.dumps({k: {n: v for n, v in value.items() if n in
          ("short_answer.rougeL", "answer_verification.correct", "paraphrase.answer_preserved")}
          for k, value in level.items()}, indent=2))


if __name__ == "__main__":
    main()
