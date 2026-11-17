---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- urban-mobility
- instruction-tuning
- transportation
---
# UrbanMobility-Instruct: A Dataset-Centric Instruction Benchmark for Grounded Reasoning over Transportation Data

UrbanMobility-Instruct is an English instruction dataset for training and evaluating language models on grounded reasoning about transportation datasets. Each question supplies dataset context and asks the model to interpret it, answer a question, assess a candidate answer, or paraphrase a completed instruction while preserving its meaning.

The benchmark evaluates understanding of dataset documentation, including coverage, sampling, variables, collection methods, and limitations. It does not require downloading or processing raw vehicle trajectories or trip records.

## What is included?

The primary corpus draws on four empirical data resources and one literature-derived source. Their documentation and relevant literature supply the evidence for authored questions; the raw transportation records are not the instruction-training inputs.

| Primary source | Information represented | Spatial and temporal characteristics |
| --- | --- | --- |
| highD | Drone-recorded vehicle positions and maneuvers | Six highway locations in Germany; frame-level recordings |
| NGSIM | Video-derived vehicle trajectories and lane behavior | California road segments, including I-80 and US 101; 0.1-second intervals |
| Road Networks | Road vertices and edges | Regional networks, including California and Pennsylvania; static snapshots |
| NYC TLC Trip Records | Taxi and for-hire trip activities | New York City pickup/drop-off locations and zones; trip timestamps and monthly files |
| Urban Flow Prediction Survey | Literature-derived urban-flow concepts and methods | Studies across settings and forecast horizons; no single fixed sampling interval |

The primary sources contribute **160 authored seeds**. **Île-de-France Mobilités Open Data** and Japan's **MLIT Road Traffic Census** contribute another **66 seeds**, reserved entirely for held-out-source evaluation. A separate external benchmark adds **350 independently authored questions from ten additional sources**.

## Reasoning levels and question formats

Each seed contains a supporting source fact, question, reference answer, explanation, and reasoning level. Questions use either four-option multiple-choice (MC) or short-answer (SA) formats.

| Level | Reasoning requirement | Construction |
| --- | --- | --- |
| 1: Single-fact | Identify one documented source property | One MC and one SA seed for each of ten perspectives, giving 20 Level-1 seeds per primary source |
| 2: Multi-attribute | Relate multiple documented characteristics | Combine attributes such as coverage, measurement resolution, and variable definitions |
| 3: Contextual | Interpret the analytical implications of source characteristics | Five SA seeds each for TLC and the Urban Flow Prediction Survey; the other primary sources have no Level-3 seeds |

The ten Level-1 perspectives are provenance, spatial coverage, temporal coverage, measurement resolution, variable definitions, file structure, collection methodology, data accessibility and availability, usage context, and dataset limitations. Level 3 concerns source-grounded analytical suitability and constraints; it does not require raw-data computation or independent planning judgment.

## Dataset construction

The construction procedure follows five stages:

1. **Define reasoning levels and instruction schemas.** Establish the MC/SA formats and required source-grounded fields.
2. **Author grounded seeds.** Link questions, answers, and explanations to documentation or literature, assigning each seed a unique `seed_id`.
3. **Establish concept groups.** Build `concept_group_id` from the source name and a hash of normalized `Fact` text. MC and SA seeds supported by the same fact can share a group. Two manually reviewed alias mappings merge equivalent pairs in TLC and Road Networks. There is no separate `concept_id` field.
4. **Generate controlled variants.** GPT-5.1 varies fact and question wording, answers, explanations, and MC option order while retaining source, seed, group, and reasoning-level lineage. Reordered options receive updated answer labels. Each variant has a distinct `variant_id`. Incorrect candidates are constructed for verification and are never desired answer-generation targets.
5. **Validate, format, and partition.** Validate canonical records, materialize the three task views, and then apply one shared resolved group-assignment map. Group identities exist before augmentation; partition assignments are applied after task-view assembly.

The resulting **14,726 canonical records** comprise **10,560 primary-source records** and **4,166 France/Japan records**. Each canonical record provides one generation example. Augmented descendants remain variants of existing seeds, not independently authored concepts.

### Primary corpus composition

The following counts cover the five primary sources before partitioning and exclude France/Japan and the external benchmark.

| Source | Level 1 | Level 2 | Level 3 | SA | MC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| highD | 1,520 | 510 | 0 | 715 | 1,315 | 2,030 |
| NGSIM | 1,520 | 510 | 0 | 715 | 1,315 | 2,030 |
| Road Networks | 1,520 | 510 | 0 | 715 | 1,315 | 2,030 |
| TLC Trip Records | 1,520 | 510 | 205 | 920 | 1,315 | 2,235 |
| Urban Flow Prediction Survey | 1,520 | 510 | 205 | 920 | 1,315 | 2,235 |
| Total | 7,600 | 2,550 | 410 | 3,985 | 6,575 | 10,560 |

### Supervision views

Validated canonical records are materialized into three views:

| Task view | Model input and expected output |
| --- | --- |
| `generation` | Dataset context and a question; produce the answer and explanation. |
| `verification` | Context, question, and candidate answer; judge whether the candidate is supported. Correct and incorrect candidates are paired. |
| `paraphrase` | A completed instruction; rewrite it while preserving facts, answers, option relationships, explanations, and field labels. |

Verification uses paired `Correct/Yes` and `Incorrect/No` targets under the `VERDICT` and `CANDIDATE MATCHES CONTEXT` labels. For paraphrase, eligible augmented descendants serve as targets for completed original instructions. Eligibility requires matching stored correct answers; it does not guarantee full semantic equivalence. These views share underlying questions and should not be counted as independent source questions.

## Splits and evaluation settings

Use the prepared files in [`data/urbanmobility_instruct_v1/core/experiment/`](data/urbanmobility_instruct_v1/core/experiment). This is version 1 of the UrbanMobility-Instruct corpus.

| Partition | Filename prefix | Generation | Verification | Paraphrase | Intended use |
| --- | --- | ---: | ---: | ---: | --- |
| Training | `train__` | 8,326 | 16,652 | 4,260 | Model fitting |
| Validation | `validation__` | 1,036 | 2,072 | 560 | Model selection |
| In-domain test | `test__` | 1,198 | 2,396 | 760 | Held-out concepts from the five main sources |
| France/Japan holdout | `cross_regional__` | 4,166 | 8,332 | 2,420 | Geographic transfer; evaluation only |
| Total | | 14,726 | 29,452 | 8,000 | |

The 106 resolved primary-source groups are split deterministically by source, using random seed **2025** and an **80:10:10 target ratio**: 85 training, 11 validation, and 10 test groups. Ratios apply to groups, not individual records, question formats, or reasoning levels. Related seeds and their descendants remain together across all three task views. Preserve these assignments rather than randomly splitting individual records. The assignments are recorded in [`split_manifest.json`](data/urbanmobility_instruct_v1/core/reporting/split_manifest.json).

France and Japan each contribute 33 authored seeds and use the same reasoning taxonomy, controlled augmentation, and task-view construction as the primary corpus. Both sources are excluded from training, validation, and checkpoint selection. Their task files are present for all three views, for example `experiment/generation/cross_regional__france_mobility_instructions.json` and `experiment/generation/cross_regional__japan_mlit_instructions.json` beneath the main data directory. This setting measures controlled held-out-source transfer under familiar instruction formulations.

## Quick start

Install the download client:

```bash
pip install huggingface_hub
```

If repository access requires authentication, log in with an account that has access using `hf auth login`.

The following loads all source files for one task and partition. It resolves `main` to a commit once so that every download uses the same snapshot. Save that commit ID with experiment results for reproducibility.

```python
import json
from huggingface_hub import HfApi, hf_hub_download

REPO = "UlyssesLynne/UrbanMobility-Instruct"
api = HfApi()
revision = api.dataset_info(REPO, revision="main").sha
files = api.list_repo_files(REPO, repo_type="dataset", revision=revision)
base = "data/urbanmobility_instruct_v1/core/experiment"

def load_partition(task="generation", split="test"):
    prefix = f"{base}/{task}/{split}__"
    paths = sorted(p for p in files if p.startswith(prefix) and p.endswith(".json"))
    if not paths:
        raise ValueError(f"No files found for {task=}, {split=}")
    records = []
    for filename in paths:
        path = hf_hub_download(REPO, filename, repo_type="dataset", revision=revision)
        with open(path, encoding="utf-8") as stream:
            records.extend(json.load(stream)["records"])
    return records

test = load_partition("generation", "test")
regional = load_partition("generation", "cross_regional")
assert len(test) == 1198
assert len(regional) == 4166
print("Dataset commit:", revision)
print(test[0]["prompt"])
print(test[0]["target"])
```

Use `train` or `validation` for the development partitions, and `verification` or `paraphrase` for the other task views. Feed `prompt` to the model and use `target` as the reference output.

## Record format

Prepared partition files are JSON objects containing a `records` array, not JSON Lines files. Their top-level metadata includes `schema_version`, `source_name`, `split`, and `region`.

| Record field | Meaning |
| --- | --- |
| `example_id` | Identifier for a task example |
| `seed_id`, `variant_id` | Original question and derived variant identifiers |
| `concept_group_id` | Group used to keep related examples in one partition |
| `source_name` | Source dataset identifier |
| `task_type` | Question or task format |
| `Level` | Reasoning level, from 1 to 3 |
| `prompt`, `target` | Model input and reference output |
| `split`, `evaluation_scope` | Partition and evaluation setting |

Verification records also contain `base_task_type` and `candidate_polarity`. Paraphrase records include `base_task_type` and `source_variant_id`. Preserve the target formatting when reproducing the associated evaluations.

## Independent ten-region test benchmark

The external benchmark is included in [`data/urbanmobility_instruct_v1/external_benchmark/`](data/urbanmobility_instruct_v1/external_benchmark). It contains **350 authored questions: 35 per source**, with source profiles and a seed manifest. All ten sources are reserved for evaluation:

| Location | Source |
| --- | --- |
| Melbourne, Australia | Melbourne Pedestrian Counting System |
| São Paulo, Brazil | São Paulo Origin-Destination Survey 2017 |
| Bogotá, Colombia | Bogotá TransMilenio and SITP GTFS |
| Accra, Ghana | Accra GTFS |
| Athens, Greece | Athens Bus and Trolley Routes |
| Iceland | Vegagerðin Traffic Statistics |
| Nairobi, Kenya | Nairobi Digital Matatus |
| Auckland, New Zealand | Auckland Monthly Cycle Monitoring |
| Madrid, Spain | Madrid Traffic Accidents |
| Stockholm, Sweden | Stockholm Open Traffic Data |

Unlike the primary and France/Japan corpora, this benchmark is authored directly from documented source information without large-scale augmentation. Its 350 questions comprise **150 MC and 200 SA questions**, producing **350 generation examples and 700 balanced verification instances** (35 and 70 per source, respectively). No paraphrase view is generated. None of these sources contributes records to training or checkpoint selection.

Together, the three evaluation settings distinguish held-out source-fact groups within the primary corpus, new sources under controlled augmentation, and independently authored questions from additional datasets.

**Packaging:** this folder contains the source question files in `seeds/`, `source_profiles.json`, and `seed_manifest.json`. It does not currently contain prebuilt `generation/` or `verification/` task files. The generation and verification counts above describe the benchmark constructed from these seeds, not additional task files already present in this repository.

To inspect the external questions directly, run this after the quick-start setup:

```python
external_questions = []
for filename in sorted(files):
    if not filename.startswith("data/urbanmobility_instruct_v1/external_benchmark/seeds/"):
        continue
    if not filename.endswith(".json"):
        continue
    path = hf_hub_download(REPO, filename, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as stream:
        source = json.load(stream)
    for instruction in source["instructions"]:
        for question in instruction.get("Positive Example", []):
            external_questions.append({"source_file": filename, **question})

assert len(external_questions) == 350
print(external_questions[0]["Fact"])
print(external_questions[0]["Question"])
print(external_questions[0]["Output"])
```

Seed questions use fields such as `Fact`, `Question`, `Output`, `Explanation`, and `Level`, with `Selections` for multiple-choice questions. They use a different schema from the prepared main-corpus task files. Source profiles provide documentation and provenance for the external sources.

## Repository guide

| Path | Contents |
| --- | --- |
| `data/urbanmobility_instruct_v1/core/experiment/` | Prepared training, validation, in-domain test, and France/Japan task files; start here |
| `data/urbanmobility_instruct_v1/core/canonical/` | Canonical question records before task formatting |
| `data/urbanmobility_instruct_v1/core/reporting/` | Split manifest, statistics, validation and leakage audits |
| `data/urbanmobility_instruct_v1/seeds/` | Original questions for the main corpus and France/Japan |
| `data/urbanmobility_instruct_v1/external_benchmark/` | Ten-source external test questions and source metadata |
| `dataset_manifest.json` | Machine-readable provenance, data hashes, and associated model/evaluation paths |

Other folders within `core/` retain intermediate and unsplit task views for traceability. Use the prepared `experiment/` files for training and evaluation.

## Construction, quality, and limitations

Automated validation checks required fields and identifiers, MC answer-label consistency, exact duplicates, separation of generation prompts and targets, and correct–incorrect candidate distinctness. All **14,726 canonical records** pass the final structural gate, with **zero rejections and no exact duplicates**. A lexical-overlap heuristic retains **4,260 warnings** about answer–fact correspondence rather than automatically rejecting potentially valid reformulations. Grounding is established during seed authoring; structural checks do not independently establish factual correctness.

Deterministic audits find zero cross-partition overlap in resolved groups, seeds, or normalized exact generation prompts in the primary train/validation/test sets. A complementary `all-mpnet-base-v2` semantic screen, with cosine similarity at least 0.90, flags three unique concept pairs. Manual review judges them related but distinct because they require different answers or reasoning operations, yielding zero confirmed leakage cases among the reviewed pairs. This semantic screen excludes France/Japan and does not rule out all residual conceptual similarity. Audit reports are included under `reporting/`.

The in-domain test contains only ten independent concept groups despite its larger record count. France/Japan use the familiar task interface and controlled augmentation, so strong results there should not be interpreted as unrestricted geographic generalization. Report the independent ten-source benchmark separately. Verification inputs expose candidate answers and paraphrase inputs expose completed answers; these can provide answer cues. Paraphrase eligibility is not a semantic-equivalence guarantee. This resource evaluates interpretation of supplied text, rather than operational transport forecasting or decisions about individual travelers.

## License and related resources

This repository declares the MIT license. Terms for the underlying source datasets remain applicable; this repository does not grant additional rights to their raw data.

- [Code and experiment notebooks](https://github.com/YufeiLinUlysses/LLM_Urban_Planning)
- [Trained models](https://huggingface.co/UlyssesLynne/urban-planning-llm-model-zoo-v3)
- [Predictions and metrics](https://huggingface.co/datasets/UlyssesLynne/urban-planning-llm-predictions-v3)

When reporting use of this dataset, include its name, repository URL, the downloaded commit ID, and the task views and evaluation partitions used.
