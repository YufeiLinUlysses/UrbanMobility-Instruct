"""Build the corrected-corpus Qwen2.5-14B Colab rerun notebook."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/urban_science_qwen25_7b_prompt_cleanup_rerun.ipynb"
OUTPUT = ROOT / "notebooks/urban_science_qwen25_14b_prompt_cleanup_rerun.ipynb"


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    source = json.dumps(notebook, ensure_ascii=False)
    for old, new, count in (
        ("Qwen2.5-7B", "Qwen2.5-14B", 6),
        ("qwen25_7b", "qwen25_14b", 1),
        ("revision-v6-qwen25-7b-prompt-cleaned-v1", "revision-v6-qwen25-14b-prompt-cleaned-v1", 1),
        ("learning_rate: float = 1e-4", "learning_rate: float = 5e-5", 1),
    ):
        actual = source.count(old)
        if actual != count:
            raise RuntimeError(f"Template drift for {old!r}: expected {count}, found {actual}")
        source = source.replace(old, new)
    notebook = json.loads(source)
    config_cell = notebook["cells"][4]
    assert config_cell["cell_type"] == "code"
    config = "".join(config_cell["source"])
    assert config.count("email_notifications: bool = False") == 1
    config = config.replace(
        "email_notifications: bool = False",
        "email_notifications: bool = True\n    auto_unassign: bool = True",
    )
    config_cell["source"] = config.splitlines(True)

    credentials_markdown = notebook["cells"][5]
    assert credentials_markdown["cell_type"] == "markdown"
    credentials_markdown["source"] = (
        "Add `HF_TOKEN`, `GMAIL_ADDRESS`, and `GMAIL_TOKEN` to Colab Secrets. "
        "`GMAIL_TOKEN` is a Gmail app password; `NOTIFY_EMAIL` is optional and "
        "defaults to `GMAIL_ADDRESS`. The notebook sends training and final "
        "verification emails, then releases the Colab runtime.\n"
    ).splitlines(True)

    credentials_cell = notebook["cells"][6]
    assert credentials_cell["cell_type"] == "code"
    credentials = "".join(credentials_cell["source"])
    marker = 'print("HF credential available; value not printed.")'
    assert credentials.count(marker) == 1
    credentials = credentials.replace(marker, '''if CFG.email_notifications:
    missing_email_secrets = [name for name in ("GMAIL_ADDRESS", "GMAIL_TOKEN") if not secret(name)]
    if missing_email_secrets:
        raise RuntimeError(f"Email notifications require Colab Secrets: {missing_email_secrets}")
print("HF credential available; email notifications configured; values not printed.")''')
    credentials_cell["source"] = credentials.splitlines(True)

    notebook["cells"].extend([
        {"cell_type": "markdown", "metadata": {}, "source": [
            "## 11. Release the Colab runtime\n",
            "Run this cell after the checkpoint and all six evaluation bundles have been verified. "
            "With `CFG.auto_unassign=True`, it returns the GPU to Colab.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": [
            "if CFG.auto_unassign:\n",
            "    from google.colab import runtime\n",
            "    print('All Qwen2.5-14B artifacts verified and completion email sent; releasing runtime.')\n",
            "    runtime.unassign()\n",
        ]},
    ])
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
