import pytest

from agents.analyst import AnalystAgent
from schemas import ProductSpecs


@pytest.mark.parametrize("image_path", ["tests/test_image.png"])
def test_analyst_returns_product_specs(image_path):
    analyst = AnalystAgent()
    result = analyst.analyse(image_path)

    assert isinstance(result, ProductSpecs)
