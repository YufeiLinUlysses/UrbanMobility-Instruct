import pytest

from urban_mobility.pipelines.data_augmentation.llm import OpenAIAugmenter
from urban_mobility.pipelines.data_augmentation.nodes import _record_errors
from urban_mobility.pipelines.data_augmentation.question_content import question_content_errors
from urban_mobility.pipelines.data_augmentation.schemas import QuestionList


@pytest.mark.parametrize('block', [
    'Correct option:\nB', 'correct answer: hourly', 'Correct choice: C',
    'Answer key: B', 'ANSWER:\nB', 'Options:\n{"A": "Hourly", "B": "Daily"}',
])
def test_rejects_embedded_blocks_in_stem(block):
    assert question_content_errors('What is the sampling interval?\n\n' + block)


def test_allows_answer_related_words_in_a_real_question():
    assert question_content_errors('Which answer correctly describes the available options for analysis?') == []


def test_cached_question_response_is_validated(monkeypatch):
    # _parse returns cached and fresh responses through the same interface.
    augmenter = object.__new__(OpenAIAugmenter)
    monkeypatch.setattr(augmenter, '_parse', lambda **kwargs: QuestionList(
        questions=['What is the interval?\n\nCorrect option:\nB']))
    with pytest.raises(ValueError, match='answer_key_in_question'):
        augmenter.questions('system', 'user', 1, temperature=0.7)


def test_structural_gate_rejects_contaminated_question_but_accepts_intended_target():
    r = dict(seed_id='s', variant_id='v', task_type='multiple_choice',
             Fact='Samples are collected hourly.', Question='What is the interval?',
             correct_answer='B', correct_answer_text='Hourly', incorrect_candidate='Daily',
             Explanation='The context states an hourly interval.', Selections={'A': 'Daily', 'B': 'Hourly'})
    assert _record_errors(r)[0] == []
    r['Question'] += '\n\nCorrect option:\nB'
    assert 'answer_key_in_question' in _record_errors(r)[0]
