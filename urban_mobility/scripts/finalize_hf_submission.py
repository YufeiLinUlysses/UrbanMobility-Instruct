"""Audit and optionally publish submission cards/indexes without retraining models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/revisions/revision_v6_prompt_cleaned"
OUT = ROOT / "data/08_reporting/submission_release"
DATA_REPO = "UlyssesLynne/urban_planning_llm"
MODEL_REPO = "UlyssesLynne/urban-planning-llm-model-zoo-v3"
PRED_REPO = "UlyssesLynne/urban-planning-llm-predictions-v3"
PIN = "775783b5b652a710182b3f1a871e231c5516d942"
TAG = "submission-v1"
PREFIX = "data/revision_v6_prompt_cleaned/"
RUNS = {
    "t5_base": "revision-v6-t5-prompt-cleaned-v1",
    "qwen25_7b": "revision-v6-qwen25-7b-prompt-cleaned-v1",
    "qwen25_14b": "revision-v6-qwen25-14b-prompt-cleaned-v1",
    "llama31_8b": "revision-v6-llama31-8b-prompt-cleaned-v1",
    "llama31_70b": "revision-v6-llama31-70b-prompt-cleaned-v1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT.parent / ".env")
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    OUT.mkdir(parents=True, exist_ok=True)
    repos = [(DATA_REPO, "dataset"), (MODEL_REPO, "model"), (PRED_REPO, "dataset")]
    legacy = [
        ("UlyssesLynne/urban-planning-llm-model-zoo", "model"),
        ("UlyssesLynne/urban-planning-llm-predictions", "dataset"),
        ("UlyssesLynne/urban-planning-llm-teacher-paraphrases", "dataset"),
    ]
    if args.publish:
        for repo, kind in repos:
            if any(ref.name == TAG for ref in api.list_repo_refs(repo, repo_type=kind).tags):
                raise RuntimeError(
                    f"{repo} already has {TAG}; refusing to overwrite a finalized release"
                )
    heads = {repo: api.repo_info(repo, repo_type=kind).sha for repo, kind in repos + legacy}
    trees = {
        repo: {
            e.path: e
            for e in api.list_repo_tree(
                repo,
                repo_type=kind,
                revision=PIN if repo == DATA_REPO else heads[repo],
                recursive=True,
            )
            if hasattr(e, "blob_id")
        }
        for repo, kind in repos
    }

    def read(repo, kind, path):
        revision = PIN if repo == DATA_REPO else heads[repo]
        return json.loads(
            Path(
                hf_hub_download(repo, path, repo_type=kind, revision=revision, token=token)
            ).read_text(encoding="utf-8")
        )

    hashes = {}
    for p in sorted(DATA.rglob("*.json")):
        path = PREFIX + p.relative_to(DATA).as_posix()
        raw = p.read_bytes()
        entry = trees[DATA_REPO][path]
        digest = hashlib.sha256(raw).hexdigest()
        observed = (
            digest
            if entry.lfs
            else hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        )
        assert observed == (entry.lfs.sha256 if entry.lfs else entry.blob_id), path
        hashes[path] = digest
    assert len(hashes) == 112
    split = json.loads((DATA / "reporting/split_manifest.json").read_text())
    split_hash = hashlib.sha256(
        json.dumps(split, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    assert (
        json.loads((DATA / "reporting/final_verification.json").read_text())[
            "residual_embedded_blocks"
        ]
        == 0
    )
    assert json.loads((DATA / "reporting/leakage_audit.json").read_text())["passed"]

    models = {}
    for key, run in RUNS.items():
        prefix = f"{key}/{run}/checkpoint"
        manifest = read(MODEL_REPO, "model", prefix + "/training_manifest.json")
        assert manifest["training_run_id"] == run and manifest["model_key"] == key
        assert manifest["dataset_version"] == "revision_v6_prompt_cleaned"
        assert manifest["split_manifest_hash"] == split_hash
        assert set(manifest["tasks"]) == {"generation", "verification", "paraphrase"}
        required = [
            "training_manifest.json",
            "training_history.json",
            "tokenizer.json",
            "model.safetensors" if key == "t5_base" else "adapter_model.safetensors",
            "config.json" if key == "t5_base" else "adapter_config.json",
        ]
        for name in required:
            assert trees[MODEL_REPO][prefix + "/" + name].size > 0
        models[key] = {
            "run_id": run,
            "checkpoint_subfolder": prefix,
            "base_model_id": manifest["base_model_id"],
            "checkpoint_type": "full_model" if key == "t5_base" else "peft_adapter",
        }

    jobs = []
    for key, run in RUNS.items():
        for scope in ["in_domain", "cross_regional", "multi_regional"]:
            version = (
                "multi_regional_external_v1"
                if scope == "multi_regional"
                else "revision_v6_prompt_cleaned"
            )
            for stage in ["base", "fine_tuned"]:
                label = "base" if stage == "base" else "finetuned"
                path = f"evaluations/{version}/{key}/{stage}/{scope}/{run}-{label}-{scope.replace('_', '-')}"
                for name in [
                    "evaluation_manifest.json",
                    "metrics.json",
                    "predictions.parquet",
                    "grouped_metrics.parquet",
                    "review_queue.parquet",
                ]:
                    assert path + "/" + name in trees[PRED_REPO], path + "/" + name
                jobs.append((key, stage, scope, version, path))

    def check(job):
        key, stage, scope, version, path = job
        manifest = read(PRED_REPO, "dataset", path + "/evaluation_manifest.json")
        for field, value in [
            ("model_key", key),
            ("checkpoint_stage", stage),
            ("dataset_scope", scope),
            ("dataset_version", version),
        ]:
            assert manifest[field] == value, path
        assert manifest["prediction_count"] > 0
        if stage == "fine_tuned":
            assert manifest["checkpoint_uri"] == MODEL_REPO
            assert manifest["checkpoint_subfolder"] == models[key]["checkpoint_subfolder"]
        return {
            "model_key": key,
            "stage": stage,
            "scope": scope,
            "path": path,
            "prediction_count": manifest["prediction_count"],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        evaluations = list(pool.map(check, jobs))
    release = {
        "release": TAG,
        "dataset_version": "revision_v6_prompt_cleaned",
        "dataset_repo": DATA_REPO,
        "dataset_source_revision": PIN,
        "dataset_subdir": PREFIX.rstrip("/"),
        "split_manifest_hash": split_hash,
        "model_repo": MODEL_REPO,
        "model_source_revision": heads[MODEL_REPO],
        "prediction_repo": PRED_REPO,
        "prediction_source_revision": heads[PRED_REPO],
        "models": models,
        "evaluations": evaluations,
        "dataset_sha256": hashes,
        "verification": "112 dataset JSON content hashes; five checkpoint manifests and required file presence; 30 evaluation manifests and required file presence. No retraining or full weight/prediction download performed.",
    }
    raw = (json.dumps(release, indent=2) + "\n").encode()
    (OUT / "submission_release.json").write_bytes(raw)
    title = "UrbanMobility-Instruct: A Dataset-Centric Instruction Benchmark for Grounded Reasoning over Transportation Data"
    common = f"Release **{TAG}** preserves the corrected `revision_v6_prompt_cleaned` data and original run IDs. See `submission_release.json` for exact paths, source commits, and dataset SHA-256 hashes.\n\n"
    links = f"[Dataset](https://huggingface.co/datasets/{DATA_REPO}) · [Models](https://huggingface.co/{MODEL_REPO}) · [Evaluations](https://huggingface.co/datasets/{PRED_REPO})\n"
    cards = {}
    cards[DATA_REPO] = (
        f"---\nlicense: mit\ntask_categories:\n- question-answering\nlanguage:\n- en\ntags:\n- urban-mobility\n- instruction-tuning\n- transportation\n---\n# {title}\n\n"
        + common
        + "## Data\n\nUse `data/revision_v6_prompt_cleaned/experiment/{generation,verification,paraphrase}/` together with `reporting/split_manifest.json`. JSON partitions contain a `records` list. The original immutable source commit is `"
        + PIN
        + "`.\n\n| Generation split | Records |\n| --- | ---: |\n| Train | 8,326 |\n| Validation | 1,036 |\n| Test | 1,198 |\n| France/Japan holdout | 4,166 |\n\nThe full task views contain 14,726 generation, 29,452 verification, and 8,000 paraphrase records. Splits preserve concept groups. An additional ten-source external benchmark is supplied under `data/multi_regional_external_v1/`; it is excluded from training. Original seeds are in `data/seed_instructions/`.\n\nThe seven root `*_final.json` files are historical exports, retained for compatibility; they are not the submission dataset. Use the release paths above. The saved audits report zero residual embedded question blocks and a passing deterministic split audit; this is not a claim that every generated fact was independently verified. Source-dataset terms still apply.\n\n"
        + links
    )
    rows = "\n".join(
        f"| {key} | `{item['base_model_id']}` | [{item['checkpoint_type']}]({item['checkpoint_subfolder']}) |"
        for key, item in models.items()
    )
    cards[MODEL_REPO] = (
        "---\nlanguage:\n- en\ntags:\n- urban-mobility\n- peft\n---\n# UrbanMobility-Instruct — model checkpoints\n\n"
        + common
        + "| Model | Actual training base | Checkpoint |\n| --- | --- | --- |\n"
        + rows
        + "\n\nT5 is a full model; the four causal checkpoints are PEFT adapters requiring their recorded base model. Load from the listed subfolder, not the repository root. The 70B run uses the Unsloth 4-bit base shown above. All five training manifests match the corrected dataset's split hash. Earlier runs remain historical; select the exact submission paths in the index. Base-model licenses and access requirements continue to apply.\n\n"
        + links
    )
    cards[PRED_REPO] = (
        "---\nlanguage:\n- en\ntags:\n- urban-mobility\n- evaluation\n---\n# UrbanMobility-Instruct — evaluation artifacts\n\n"
        + common
        + "The index selects 30 runs: five models × base/fine-tuned × in-domain/France–Japan/multi-regional evaluation. Each contains predictions, metrics, grouped metrics, a review queue, and an evaluation manifest. Corrected primary evaluations use `evaluations/revision_v6_prompt_cleaned/`; external evaluations use `evaluations/multi_regional_external_v1/`. Select the exact run IDs in the index, since historical evaluations are retained. Automated reference-overlap and entailment metrics are not independent factual correctness labels.\n\n"
        + links
    )
    operations = {}
    for repo, kind in repos:
        operations[repo] = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=cards[repo].encode()),
            CommitOperationAdd(path_in_repo="submission_release.json", path_or_fileobj=raw),
        ]
        (OUT / (repo.split("/")[-1] + "_README.md")).write_text(cards[repo], encoding="utf-8")
    for path in trees[DATA_REPO]:
        if path.startswith(PREFIX) and path.endswith(".json"):
            operations[DATA_REPO].append(
                CommitOperationCopy(src_path_in_repo=path, path_in_repo=path, src_revision=PIN)
            )
    operations[DATA_REPO].append(
        CommitOperationAdd(
            path_in_repo=PREFIX + "README.md",
            path_or_fileobj=(
                "# Corrected submission data\n\n"
                + common
                + "Refer to the repository dataset card and preserved reporting files for counts, correction history, and split assignments.\n"
            ).encode(),
        )
    )
    for folder, remote in [
        (ROOT / "data/01_raw/seed_instructions", "data/seed_instructions"),
        (ROOT / "data/01_raw/multi_regional_external_v1", "data/multi_regional_external_v1"),
    ]:
        for p in sorted(folder.rglob("*.json")):
            operations[DATA_REPO].append(
                CommitOperationAdd(
                    path_in_repo=remote + "/" + p.relative_to(folder).as_posix(), path_or_fileobj=p
                )
            )
    print(
        f"Audit passed: {len(hashes)} data files, {len(models)} models, {len(evaluations)} evaluations.",
        flush=True,
    )
    if not args.publish:
        print("Prepared cards and release index. Use --publish to update repositories.")
        return
    results = {}
    for repo, kind in repos:
        api.create_tag(
            repo,
            repo_type=kind,
            tag="pre-submission-cleanup-2026-10-14",
            revision=heads[repo],
            exist_ok=True,
        )
        commit = api.create_commit(
            repo,
            repo_type=kind,
            operations=operations[repo],
            parent_commit=heads[repo],
            commit_message="Finalize UrbanMobility-Instruct submission release and artifact index",
        )
        api.create_tag(
            repo,
            repo_type=kind,
            tag=TAG,
            revision=commit.oid,
            tag_message="Verified submission artifact selection",
        )
        results[repo] = commit.oid
        print("Published", repo, commit.oid, flush=True)
    for repo, kind in legacy:
        files = api.list_repo_files(repo, repo_type=kind, revision=heads[repo])
        old = ""
        frontmatter = "---\ntags:\n- historical\n---\n"
        if "README.md" in files:
            old = Path(
                hf_hub_download(
                    repo, "README.md", repo_type=kind, revision=heads[repo], token=token
                )
            ).read_text(encoding="utf-8")
            if old.startswith("---"):
                end = old.find("\n---", 3)
                if end >= 0:
                    frontmatter = old[: end + 4] + "\n"
                    old = old[end + 4 :].lstrip()
        card = (
            frontmatter
            + "# Historical Urban Planning artifacts\n\nThis repository is retained for provenance and is not the finalized UrbanMobility-Instruct submission release. Use the current repositories below and their `submission-v1` tags.\n\n"
            + links
            + "\n## Original documentation\n\n"
            + old
        )
        api.create_tag(
            repo,
            repo_type=kind,
            tag="pre-submission-cleanup-2026-10-14",
            revision=heads[repo],
            exist_ok=True,
        )
        commit = api.create_commit(
            repo,
            repo_type=kind,
            parent_commit=heads[repo],
            operations=[
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())
            ],
            commit_message="Mark legacy artifacts historical and link finalized submission release",
        )
        results[repo] = commit.oid
        print("Marked historical", repo, flush=True)
    (OUT / "published_commits.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
