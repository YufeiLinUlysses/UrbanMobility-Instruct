"""Compare published Qwen2.5-7B revision-v5 and prompt-cleaned runs."""
from pathlib import Path
import json

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/08_reporting/qwen7_prompt_cleanup_comparison_2026-09-15"
REPO = "UlyssesLynne/urban-planning-llm-predictions-v3"
MODEL_KEY = "qwen25_7b"
OLD_RUN = "revision-v5-qwen25-7b-three-task-chat-v2"
NEW_RUN = "revision-v6-qwen25-7b-prompt-cleaned-v1"

METRICS = {
    "MC normalized accuracy (%)": ("mc_answer", "normalized_correct", 100),
    "MC strict accuracy (%)": ("mc_answer", "strict_correct", 100),
    "MC explanation BERTScore": ("mc_explanation", "bertscore_f1", 1),
    "SA exact match (%)": ("short_answer", "exact_match", 100),
    "SA ROUGE-L": ("short_answer", "rougeL", 1),
    "SA BERTScore F1": ("short_answer", "bertscore_f1", 1),
    "SA NLI mean": ("short_answer", "nli_entailment_score", 1),
    "Verification accuracy (%)": ("answer_verification", "correct", 100),
    "Paraphrase BERTScore": ("paraphrase", "bertscore_f1", 1),
    "Paraphrase answer preserved (%)": ("paraphrase", "answer_preserved", 100),
    "Paraphrase field preservation (%)": ("paraphrase", "field_preservation", 100),
}


def prefix(scope: str, stage: str, version: str) -> str:
    run = OLD_RUN if version == "old" else NEW_RUN
    dataset = "multi_regional_external_v1" if scope == "multi_regional" else (
        "revision_v5" if version == "old" else "revision_v6_prompt_cleaned"
    )
    label = "base" if stage == "base" else "finetuned"
    run_id = f"{run}-{label}-{scope.replace('_', '-')}"
    return f"evaluations/{dataset}/{MODEL_KEY}/{stage}/{scope}/{run_id}"


def fetch(scope: str, stage: str, version: str) -> pd.DataFrame:
    filename = prefix(scope, stage, version) + "/predictions.parquet"
    path = hf_hub_download(REPO, filename, repo_type="dataset")
    return pd.read_parquet(path)


def paired(df: pd.DataFrame) -> float:
    rows = df[df.evaluation_task.eq("answer_verification")]
    ok = rows["correct"].fillna(False) & rows["candidate_match_correct"].fillna(False) & rows["verdict_match_consistent"].fillna(False)
    return float(ok.groupby(rows["variant_id"]).all().mean() * 100)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    result = {}
    for name, (task, field, scale) in METRICS.items():
        rows = df[df.evaluation_task.eq(task)]
        if len(rows) and field in rows:
            result[name] = float(rows[field].mean() * scale)
    result["Verification paired consistency (%)"] = paired(df)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results, identities = [], []
    for scope in ("in_domain", "cross_regional", "multi_regional"):
        for stage in ("base", "fine_tuned"):
            old, new = fetch(scope, stage, "old"), fetch(scope, stage, "new")
            a, b = summarize(old), summarize(new)
            for metric in sorted(set(a) | set(b)):
                results.append({
                    "scope": scope, "stage": stage, "metric": metric,
                    "revision_v5": a.get(metric), "corrected": b.get(metric),
                    "change": b[metric] - a[metric] if metric in a and metric in b else None,
                })
            keys = ["example_id", "evaluation_task"]
            joined = old[keys + ["prompt", "raw_prediction"]].merge(
                new[keys + ["prompt", "raw_prediction"]],
                on=keys, suffixes=("_old", "_new"), validate="one_to_one",
            )
            same_prompt = joined.prompt_old.eq(joined.prompt_new)
            identities.append({
                "scope": scope, "stage": stage,
                "old_rows": len(old), "new_rows": len(new), "common_rows": len(joined),
                "changed_prompts": int((~same_prompt).sum()),
                "same_prediction_on_unchanged_prompt_pct": float(
                    joined.loc[same_prompt, "raw_prediction_old"].eq(
                        joined.loc[same_prompt, "raw_prediction_new"]
                    ).mean() * 100
                ),
            })
    table = pd.DataFrame(results)
    table.to_csv(OUT / "metric_comparison.csv", index=False)
    (OUT / "comparison.json").write_text(json.dumps({
        "old_run": OLD_RUN, "new_run": NEW_RUN,
        "metric_rows": results, "prediction_identity": identities,
    }, indent=2), encoding="utf-8")
    headline = table[table.metric.isin([
        "MC normalized accuracy (%)", "SA ROUGE-L", "Verification accuracy (%)",
        "Verification paired consistency (%)", "Paraphrase BERTScore",
        "Paraphrase answer preserved (%)",
    ])]
    print(headline.to_string(index=False))
    print("\nPrediction identity:", json.dumps(identities, indent=2))


if __name__ == "__main__":
    main()
