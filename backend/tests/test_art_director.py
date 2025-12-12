import pytest
from unittest.mock import patch

from agents.art_director import DirectorAgent
from schemas import ProductSpecs, ScenePlan


@pytest.fixture
def director():
    return DirectorAgent()


@pytest.fixture
def product_specs():
    return ProductSpecs(image_path="tests/test_image.png")


@pytest.fixture
def fake_scene_json():
    """
    Mock output that matches ScenePlan schema.
    """
    return """
    {
        "prompt": "A cinematic macro shot of the ring.",
        "negative_prompt": "extra jewelry, distortions, reflections",
        "lighting_map": {
            "source_direction": "top-right",
            "temperature": "5500K"
        },
        "inpaint_coordinates": [10, 20, 30, 40]
    }
    """


def test_director_creates_scene(director, product_specs, fake_scene_json):
    with patch.object(
        director.model, "invoke_text",
        return_value=fake_scene_json
    ):
        scene = director.create_scene(product_specs)

    assert isinstance(scene, ScenePlan)
    assert scene.prompt
    assert scene.lighting_map.source_direction
    assert isinstance(scene.inpaint_coordinates, list)
    assert len(scene.inpaint_coordinates) == 4
