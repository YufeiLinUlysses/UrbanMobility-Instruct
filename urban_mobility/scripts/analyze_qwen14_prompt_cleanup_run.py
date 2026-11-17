"""Compare the published final Qwen2.5-14B runs on paired records."""
from pathlib import Path

import analyze_qwen7_prompt_cleanup_run as comparison


comparison.MODEL_KEY = "qwen25_14b"
comparison.OLD_RUN = "revision-v5-qwen25-14b-three-task-chat-v2"
comparison.NEW_RUN = "revision-v6-qwen25-14b-prompt-cleaned-v1"
comparison.OUT = Path(__file__).resolve().parents[1] / "data/08_reporting/qwen14_prompt_cleanup_comparison_2026-09-15"


if __name__ == "__main__":
    comparison.main()
