from unittest.mock import patch
from pathlib import Path

import pytest

from agents.producer import ProducerAgent
from schemas import ScenePlan, LightingMap


@pytest.fixture
def producer():
    return ProducerAgent()


@pytest.fixture
def scene_plan():
    return ScenePlan(
        prompt="A beautiful luxury scene",
        negative_prompt="extra jewellery",
        lighting_map=LightingMap(
            source_direction="top-right",
            temperature="5500K"
        ),
        inpaint_coordinates=[10, 20, 30, 40]
    )


@pytest.fixture
def fake_bytes():
    return b"FAKE_IMAGE_BYTES"


def test_generate_scene_base(producer, scene_plan):
    with patch.object(producer.client, "invoke_text", return_value="SCENE_BASE") as mock_call:
        result = producer.generate_scene_base(scene_plan)

    assert isinstance(result, bytes)
    assert b"SCENE_BASE" in result
    mock_call.assert_called_once()


def test_inpaint_product(producer, scene_plan, fake_bytes):
    with patch.object(producer.client, "load_image", return_value=b"IMAGE_PART") as mock_load, \
         patch.object(producer.client, "invoke_with_image", return_value="INPAINTED") as mock_inpaint:

        result = producer.inpaint_product(
            base_scene_bytes=fake_bytes,
            product_png_path="tests/test_image.png",
            scene_plan=scene_plan
        )

        assert isinstance(result, bytes)
        assert b"INPAINTED" in result

        mock_load.assert_called_once()
        mock_inpaint.assert_called_once()


def test_generate_final_candidate_no_feedback(producer, scene_plan):
    with patch.object(producer, "generate_scene_base", return_value=b"BASE"), \
         patch.object(producer, "inpaint_product", return_value=b"CANDIDATE") as mock_inpaint:

        result = producer.generate_final_candidate(
            product_png_path="tests/test_image.png",
            scene_plan=scene_plan,
            feedback=None
        )

    assert isinstance(result, bytes)
    assert result == b"CANDIDATE"

    assert mock_inpaint.call_count == producer._max_retries


def test_generate_final_candidate_with_good_feedback(producer, scene_plan):
    with patch.object(producer, "generate_scene_base", return_value=b"BASE"), \
         patch.object(producer, "inpaint_product", return_value=b"CANDIDATE") as mock_inpaint:

        feedback = {"score": 95, "feedback": "Looks good"}

        result = producer.generate_final_candidate(
            product_png_path="tests/test_image.png",
            scene_plan=scene_plan,
            feedback=feedback
        )

    assert isinstance(result, bytes)
    assert result == b"CANDIDATE"
    assert mock_inpaint.call_count == 1


def test_save_candidates(producer, tmp_path, fake_bytes):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)

    result_path = producer.save_candidates(fake_bytes, str(output_dir))

    saved = Path(result_path)
    assert saved.exists()
    assert saved.read_bytes() == fake_bytes
