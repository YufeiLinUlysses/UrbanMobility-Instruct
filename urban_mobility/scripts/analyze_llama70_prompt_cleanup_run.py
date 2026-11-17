"""Compare the published final Llama 3.1-70B runs on paired records."""
from pathlib import Path

import analyze_qwen7_prompt_cleanup_run as comparison


comparison.MODEL_KEY = "llama31_70b"
comparison.OLD_RUN = "revision-v5-llama31-70b-three-task-chat-v2"
comparison.NEW_RUN = "revision-v6-llama31-70b-prompt-cleaned-v1"
comparison.OUT = Path(__file__).resolve().parents[1] / "data/08_reporting/llama70_prompt_cleanup_comparison_2026-09-17"


if __name__ == "__main__":
    comparison.main()
