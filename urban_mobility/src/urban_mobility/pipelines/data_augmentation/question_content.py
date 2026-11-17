"""Validate question stems independently of intentionally completed task inputs."""

import re

_ANSWER_KEY = re.compile(
    r"\b(?:correct\s+(?:option|answer|choice)|ground[- ]truth\s+answer|answer\s+key)\s*[:：]"
    r"|(?:^|\n)\s*(?:ANSWER|EXPLANATION|OUTPUT)\s*[:：]",
    re.IGNORECASE,
)
_OPTIONS = re.compile(r"(?:^|\n)\s*(?:Options|Selections|Choices)\s*[:：]", re.IGNORECASE)


def question_content_errors(question: str) -> list[str]:
    """Reject copied response fields/options inside a field meant to hold only a stem."""
    errors = []
    if _ANSWER_KEY.search(question):
        errors.append("answer_key_in_question")
    if _OPTIONS.search(question):
        errors.append("options_block_in_question")
    return errors


def require_question_stems(questions: list[str]) -> list[str]:
    for index, question in enumerate(questions):
        errors = question_content_errors(question)
        if errors:
            raise ValueError(f"Question variant {index} contains forbidden content: {', '.join(errors)}")
    return questions
