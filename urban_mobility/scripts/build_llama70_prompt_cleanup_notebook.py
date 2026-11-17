"""Build the corrected-corpus Llama 3.1-70B Colab rerun notebook."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/urban_science_llama31_8b_prompt_cleanup_rerun.ipynb"
OUTPUT = ROOT / "notebooks/urban_science_llama31_70b_prompt_cleanup_rerun.ipynb"
PREQUANTIZED_MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    source = json.dumps(notebook, ensure_ascii=False)
    replacements = (
        ("Llama 3.1-8B", "Llama 3.1-70B"),
        ("llama31_8b", "llama31_70b"),
        ("revision-v6-llama31-8b-prompt-cleaned-v1", "revision-v6-llama31-70b-prompt-cleaned-v1"),
        ("learning_rate: float = 1e-4", "learning_rate: float = 5e-5"),
        ("train_batch_size: int = 8", "train_batch_size: int = 6"),
        ("eval_batch_size: int = 32", "eval_batch_size: int = 4"),
        ("generation_batch_size: int = 32", "generation_batch_size: int = 8"),
        ("metric_batch_size: int = 32", "metric_batch_size: int = 8"),
        ("gradient_accumulation_steps: int = 2", "gradient_accumulation_steps: int = 3"),
        ("eval_steps: int = 50", "eval_steps: int = 25"),
        ("save_steps: int = 50", "save_steps: int = 25"),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"Template drift: {old!r}")
        source = source.replace(old, new)
    notebook = json.loads(source)
    notebook["cells"][0]["source"] = (
        "# Llama 3.1-70B corrected-corpus rerun: training and all evaluations\n\n"
        "Run top to bottom in a fresh Colab A100 80 GB runtime. This notebook trains "
        "Llama 3.1-70B-Instruct with 4-bit QLoRA on the pinned corrected "
        "`revision_v6_prompt_cleaned` partitions, then evaluates base and fine-tuned "
        "checkpoints in-domain, on France/Japan, and on the independent ten-dataset "
        "benchmark. This disk-constrained rerun loads the public prequantized "
        "`unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit` weights (about 40 GB) "
        "rather than downloading the approximately 140 GB original weights for on-load "
        "quantization. The base-weight quantization source thus differs from the older "
        "70B run and must be disclosed in any comparison. "
        "Generation and metric batches default to 8; reduce them if memory pressure occurs.\n"
    ).splitlines(True)
    notebook["cells"][2]["source"] = (
        "import subprocess\n"
        "subprocess.run(['nvidia-smi'], check=True)\n"
        "memory = subprocess.check_output([\n"
        "    'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'\n"
        "], text=True)\n"
        "total_mb = int(memory.splitlines()[0].strip())\n"
        "assert total_mb >= 75000, f'Llama 70B rerun requires an 80 GB GPU; found {total_mb} MiB'\n"
        "print(f'GPU memory: {total_mb} MiB')\n"
    ).splitlines(True)
    notebook["cells"][13]["source"] = (
        "## 6. Prepare the prequantized 70B model and check disk space\n\n"
        "The earlier smoke attempt downloaded part of the full-precision Meta model "
        "and filled Colab's disk. This step removes only that model's cache, then "
        "selects the public 4-bit weights for training and all evaluations.\n"
    ).splitlines(True)
    smoke = "".join(notebook["cells"][14]["source"])
    setup = f'''from pathlib import Path
import shutil

cache_root = (Path.home() / ".cache/huggingface/hub").resolve()
legacy_cache = (cache_root / "models--meta-llama--Meta-Llama-3.1-70B-Instruct").resolve()
assert legacy_cache.parent == cache_root
if legacy_cache.exists():
    shutil.rmtree(legacy_cache)
    print("Removed incomplete original 70B model cache.")

registry = project / "conf/base/parameters_models.yml"
registry_text = registry.read_text(encoding="utf-8")
original_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
prequantized_model = "{PREQUANTIZED_MODEL}"
assert registry_text.count(original_model) + registry_text.count(prequantized_model) == 1
registry.write_text(registry_text.replace(original_model, prequantized_model), encoding="utf-8")
free_gib = shutil.disk_usage("/content").free / 2**30
print(f"Using {{prequantized_model}}; free disk: {{free_gib:.1f}} GiB")
assert free_gib >= 55, "At least 55 GiB free disk is needed for the 4-bit model and artifacts."

'''
    notebook["cells"][14]["source"] = (setup + smoke).splitlines(True)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"]).replace(
            "All Llama 3.1-8B artifacts verified", "All Llama 3.1-70B artifacts verified"
        )
        code = code.replace(
            '"training.per_device_train_batch_size":8,"training.per_device_eval_batch_size":8,',
            '"training.per_device_train_batch_size":2,"training.per_device_eval_batch_size":2,'
        )
        cell["source"] = code.splitlines(True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
