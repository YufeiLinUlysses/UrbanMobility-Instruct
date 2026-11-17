"""Build the Colab notebook for the corrected Qwen2.5-7B rerun.

The T5 diagnostic notebook is the shared, already-tested workflow template.  This
builder changes only model-specific configuration and validates every expected
replacement so template drift cannot silently produce a mixed configuration.
"""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/urban_science_t5_prompt_cleanup_rerun.ipynb"
OUTPUT = ROOT / "notebooks/urban_science_qwen25_7b_prompt_cleanup_rerun.ipynb"


def replace_exact(text: str, old: str, new: str, expected: int) -> str:
    actual = text.count(old)
    if actual != expected:
        raise RuntimeError(
            f"Template drift for {old!r}: expected {expected} occurrence(s), found {actual}"
        )
    return text.replace(old, new)


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"Missing {TEMPLATE}. Run build_t5_prompt_cleanup_notebook.py first."
        )

    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    text = json.dumps(notebook, ensure_ascii=False)

    replacements = (
        ("# T5 corrected-corpus diagnostic: training and all evaluations",
         "# Qwen2.5-7B corrected-corpus rerun: training and all evaluations", 1),
        ("This notebook trains only T5-base", "This notebook trains Qwen2.5-7B-Instruct", 1),
        ('model_key: str = \\"t5_base\\"', 'model_key: str = \\"qwen25_7b\\"', 1),
        ('run_id: str = \\"revision-v6-t5-prompt-cleaned-v1\\"',
         'run_id: str = \\"revision-v6-qwen25-7b-prompt-cleaned-v1\\"', 1),
        ("train_batch_size: int = 32", "train_batch_size: int = 12", 1),
        ("learning_rate: float = 2e-5", "learning_rate: float = 1e-4", 1),
        ("warmup_ratio: float = 0.10", "warmup_ratio: float = 0.03", 1),
        ("weight_decay: float = 0.01", "weight_decay: float = 0.0", 1),
        ("eval_steps: int = 100", "eval_steps: int = 25", 1),
        ("save_steps: int = 100", "save_steps: int = 25", 1),
        ('training.use_chat_template\\":False', 'training.use_chat_template\\":True', 2),
        ('evaluation.use_chat_template\\":False', 'evaluation.use_chat_template\\":True', 2),
        ('training.early_stopping_patience\\":0,\\"training.random_seed\\":42',
         'training.early_stopping_patience\\":4,\\"training.early_stopping_threshold\\":0.001,\\"training.random_seed\\":42', 1),
        ("## 7. Full corrected T5 training", "## 7. Full corrected Qwen2.5-7B training", 1),
        ("T5 corrected training complete", "Qwen2.5-7B corrected training complete", 1),
        ('for name in (\\"training_manifest.json\\",\\"training_history.json\\",\\"config.json\\",\\"model.safetensors\\",\\"tokenizer.json\\"):',
         'for name in (\\"training_manifest.json\\",\\"training_history.json\\",\\"adapter_config.json\\",\\"adapter_model.safetensors\\"):', 1),
        ("T5 corrected diagnostic verified", "Qwen2.5-7B corrected rerun verified", 1),
        ("T5 corrected corpus diagnostic", "Qwen2.5-7B corrected corpus rerun", 1),
    )
    for old, new, expected in replacements:
        text = replace_exact(text, old, new, expected)

    notebook = json.loads(text)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
