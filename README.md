# UrbanMobility-Instruct

Code and data for **UrbanMobility-Instruct: A Dataset-Centric Instruction Benchmark for Grounded Reasoning over Transportation Data**.

## Submission artifacts

Use the **`submission-v1`** tags for the finalized artifact selection. Repository access is currently private and requires authorization.

| Repository | Contents |
| --- | --- |
| [Dataset](https://huggingface.co/datasets/UlyssesLynne/urban_planning_llm/tree/submission-v1) | Corrected data, preserved splits and audits, original seeds, and external benchmark seeds |
| [Models](https://huggingface.co/UlyssesLynne/urban-planning-llm-model-zoo-v3/tree/submission-v1) | Five corrected-data checkpoints: full T5 weights and four PEFT adapters |
| [Evaluations](https://huggingface.co/datasets/UlyssesLynne/urban-planning-llm-predictions-v3/tree/submission-v1) | Predictions and metrics for all 30 model/stage/evaluation-setting combinations |

Each repository includes `submission_release.json` with exact checkpoint and evaluation paths, source commits, and data hashes. On the dataset repository's `main` branch, the corpus is published under `data/urbanmobility_instruct_v1/`. Pinned historical snapshots retain the internal path `revision_v6_prompt_cleaned` for reproducibility; the seven root `*_final.json` files are historical exports. The 70B checkpoint uses `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`, as recorded in its training manifest.

## Reproduce the experiments

Use the model-specific notebooks below in a fresh **Colab GPU runtime** (or a compatible hosted Jupyter GPU environment). Each notebook installs the environment, downloads the corrected dataset, checks the splits, trains one model, and evaluates its base and fine-tuned checkpoints.

| Model | Notebook |
| --- | --- |
| T5-base | [Train and evaluate](urban_mobility/notebooks/urban_science_t5_prompt_cleanup_rerun.ipynb) |
| Qwen2.5-7B-Instruct | [Train and evaluate](urban_mobility/notebooks/urban_science_qwen25_7b_prompt_cleanup_rerun.ipynb) |
| Qwen2.5-14B-Instruct | [Train and evaluate](urban_mobility/notebooks/urban_science_qwen25_14b_prompt_cleanup_rerun.ipynb) |
| Llama3.1-8B-Instruct | [Train and evaluate](urban_mobility/notebooks/urban_science_llama31_8b_prompt_cleanup_rerun.ipynb) |
| Llama3.1-70B-Instruct | [Train and evaluate](urban_mobility/notebooks/urban_science_llama31_70b_prompt_cleanup_rerun.ipynb) |

1. Add `HF_TOKEN` to Colab Secrets. Access to the private dataset repository `UlyssesLynne/urban_planning_llm` and the selected model is required, including gated Llama access where applicable. GPU memory requirements vary; the 70B run requires substantially more capacity than the smaller models.
2. In the notebook's `Config` cell, set `git_url` and `git_branch` to a repository/branch containing this code and the `urban_mobility/` folder. Choose a unique `run_id`, and set `model_repo` and `prediction_repo` to Hugging Face repositories you can write to.
3. Keep the model-specific training settings and pinned dataset revision unchanged, then run the cells in order. The default workflow publishes checkpoints and predictions; later evaluation loads the published checkpoint. Disabling publication alone is therefore insufficient for an entirely local run.
4. Repeat for each model. Preserve the run configuration, split manifest, checkpoints, and prediction/metric files with the results.

Evaluation covers **held-out in-domain concepts**, **France/Japan cross-regional data**, and an **independent ten-dataset benchmark**. Each notebook runs base and fine-tuned evaluation across these three settings.

## Local setup and checks

With Python 3.12 and `uv` installed, run from the repository root:

```bash
cd urban_mobility
uv sync --frozen
uv run pytest
uv run kedro run --pipelines validate_inputs
```

These commands install the locked environment, run unit tests, and validate seed inputs; they do not reproduce the full GPU experiments. See the [pipeline README](urban_mobility/README.md) for individual Kedro pipelines and figure generation.

## Repository layout

- `instruction_dataset/`: preserved original seed JSON files.
- `urban_mobility/data/01_raw/seed_instructions/`: identical seed copies used by the pipeline.
- `urban_mobility/`: source code, configurations, tests, notebooks, and retained experiment data.
- `submission/`: manuscript sources and figures.

Notebook outputs are written to `Config.artifact_root` (default: `/content/urban_science_artifacts`) and published to the configured Hugging Face repositories. Download or persist local artifacts before discarding a temporary runtime.
