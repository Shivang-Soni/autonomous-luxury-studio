import pytest
from unittest.mock import MagicMock

from graph.graph_workflow import GraphWorkflow
from schemas import GraphState, ProductSpecs


@pytest.fixture
def mock_agents():
    analyst = MagicMock()
    director = MagicMock()
    producer = MagicMock()
    judge = MagicMock()

    return {
        "analyst": analyst,
        "director": director,
        "producer": producer,
        "judge": judge
    }


@pytest.fixture
def workflow(mock_agents):
    wf = GraphWorkflow(
        mock_agents["analyst"],
        mock_agents["director"],
        mock_agents["producer"],
        mock_agents["judge"]
    ).build()
    return wf


@pytest.fixture
def initial_state():
    specs = ProductSpecs(image_path="tests/test_image.png")
    return GraphState(product=specs)


def test_graph_workflow_runs_once(workflow, mock_agents, initial_state):
    # Analyst returns analysis dict
    mock_agents["analyst"].analyse.return_value = {
        "product_png_path": "tests/test_image.png"
    }

    # Director returns scene plan
    mock_agents["director"].create_scene.return_value = {"prompt": "p"}

    # Producer returns generation output
    mock_agents["producer"].generate_final_candidate.return_value = {
        "generated_image_path": "tests/test_candidate_image.png"
    }

    # Judge returns good score (workflow stops immediately)
    mock_agents["judge"].evaluate.return_value = (
        95,
        {"score": 95, "feedback": "looks good"}
    )

    final_state = workflow.invoke(initial_state)

    assert final_state.analysis is not None
    assert final_state.scene_plan is not None
    assert final_state.generation is not None
    assert final_state.judgement is not None

    mock_agents["analyst"].analyse.assert_called_once()
    mock_agents["director"].create_scene.assert_called_once()
    mock_agents["producer"].generate_final_candidate.assert_called_once()
    mock_agents["judge"].evaluate.assert_called_once()

    assert final_state.retries == 0


def test_graph_retries_until_threshold(workflow, mock_agents, initial_state):
    # Analyst always returns basic analysis
    mock_agents["analyst"].analyse.return_value = {
        "product_png_path": "tests/test_image.png"
    }

    # Director always returns same scene
    mock_agents["director"].create_scene.return_value = {"prompt": "p"}

    # Producer returns same generated image path
    mock_agents["producer"].generate_final_candidate.return_value = {
        "generated_image_path": "tests/test_candidate_image.png"
    }

    # First evaluation fails, second passes
    mock_agents["judge"].evaluate.side_effect = [
        (20, {"score": 20, "feedback": "bad"}),
        (91, {"score": 91, "feedback": "good"}),
    ]

    final_state = workflow.invoke(initial_state)

    # Exactly one retry should have happened
    assert final_state.retries == 1

    assert mock_agents["analyst"].analyse.call_count == 1
    assert mock_agents["director"].create_scene.call_count == 1

    # Producer should run twice (first try + retry)
    assert mock_agents["producer"].generate_final_candidate.call_count == 2

    # Judge should also be called twice
    assert mock_agents["judge"].evaluate.call_count == 2

    assert final_state.judgement["score"] == 91
