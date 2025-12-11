import pytest
from unittest.mock import MagicMock

from agents.judge import JudgeAgent


@pytest.fixture
def judge():
    agent = JudgeAgent()
    agent.model = MagicMock()
    return agent


def test_judge_evaluates_correctly(judge):
    # Mock returned image parts
    judge.model.load_image.return_value = {"mock": "image_part"}

    # Mock model output (strict JSON emulation)
    judge.model.invoke_with_image.return_value = \
        '{"score": 85, "feedback": "Lighting mismatch"}'

    score, feedback = judge.evaluate(
        original_image_path="tests/test_image.png",
        candidate_image_path="tests/test_candidate_image.png"
    )

    # Assertions
    assert isinstance(score, int)
    assert feedback["score"] == 85
    assert feedback["feedback"] == "Lighting mismatch"

    # Ensure correct model calls
    assert judge.model.load_image.call_count == 2
    judge.model.invoke_with_image.assert_called_once()
