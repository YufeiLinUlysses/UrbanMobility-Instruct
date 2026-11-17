"""Build the corrected-corpus Llama 3.1-8B Colab rerun notebook."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/urban_science_qwen25_14b_prompt_cleanup_rerun.ipynb"
OUTPUT = ROOT / "notebooks/urban_science_llama31_8b_prompt_cleanup_rerun.ipynb"


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    source = json.dumps(notebook, ensure_ascii=False)
    for old, new, expected in (
        ("Qwen2.5-14B", "Llama 3.1-8B", 7),
        ("qwen25_14b", "llama31_8b", 1),
        ("revision-v6-qwen25-14b-prompt-cleaned-v1", "revision-v6-llama31-8b-prompt-cleaned-v1", 1),
        ("learning_rate: float = 5e-5", "learning_rate: float = 1e-4", 1),
        ("train_batch_size: int = 12", "train_batch_size: int = 8", 1),
        ("eval_steps: int = 25", "eval_steps: int = 50", 1),
        ("save_steps: int = 25", "save_steps: int = 50", 1),
    ):
        actual = source.count(old)
        if actual != expected:
            raise RuntimeError(f"Template drift for {old!r}: expected {expected}, found {actual}")
        source = source.replace(old, new)
    notebook = json.loads(source)
    notebook["cells"][0]["source"] = (
        "# Llama 3.1-8B corrected-corpus rerun: training and all evaluations\n\n"
        "Run top to bottom in a fresh Colab A100 GPU runtime. This notebook trains "
        "Llama 3.1-8B-Instruct on the pinned corrected `revision_v6_prompt_cleaned` "
        "partitions, then evaluates base and fine-tuned checkpoints in-domain, "
        "on France/Japan, and on the independent ten-dataset benchmark. "
        "The Hugging Face token must have access to the gated Meta Llama model.\n"
    ).splitlines(True)

    config_cell = notebook["cells"][4]
    config = "".join(config_cell["source"])
    marker = "    eval_batch_size: int = 32\n"
    assert config.count(marker) == 1
    config = config.replace(
        marker,
        marker + "    generation_batch_size: int = 32\n    metric_batch_size: int = 32\n",
    )
    config_cell["source"] = config.splitlines(True)

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        code = code.replace('"evaluation.batch_size":8',
                            '"evaluation.batch_size":CFG.generation_batch_size')
        code = code.replace('"evaluation.metric_batch_size":8',
                            '"evaluation.metric_batch_size":CFG.metric_batch_size')
        cell["source"] = code.splitlines(True)
    all_code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                         if cell["cell_type"] == "code")
    assert all_code.count('"evaluation.batch_size":CFG.generation_batch_size') == 2
    assert all_code.count('"evaluation.metric_batch_size":CFG.metric_batch_size') == 2

    notebook["cells"][0]["source"].extend([
        "Evaluation uses batch 32 for generation and semantic metrics on an A100; "
        "reduce the two evaluation batch settings to 16 if a CUDA out-of-memory "
        "error occurs. Changing this file does not affect a Colab process already running.\n"
    ])
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
