from typing import Any

from llm.gemini_pipeline import GeminiClient
from schemas import ProductSpecs
from config import Configuration

config = Configuration()


class AnalystAgent:
    """
    The Gemologist Node

    Responsiblities:
    - Extracts ground truth features
      from product images to prevent hallucination.
    - Ensures accurate downstream generations.
    - Return strongly typed ProductSpecs
    """

    def __init__(self, model: str = config.ANALYST_MODEL):
        self.model = GeminiClient(model=model)

        self.system_prompt = (
            "You are a Gemologist AI specializing in jewelry. "
            "Analyze the provided product image and produce a STRICT JSON object "
            "with the following fields:\n\n"
            "{\n"
            '  "metal_type": "string",\n'
            '  "main_stone": {\n'
            '       "cut": "string",\n'
            '       "color": "string",\n'
            '       "clarity": "string",\n'
            '       "carat": "string (optional)"\n'
            "   },\n"
            '  "setting_style": "string",\n'
            '  "unique_imperfections": "string"\n'
            "}\n\n"
            "RULES:\n"
            "- The JSON MUST be valid and parseable.\n"
            "- No additional text outside the JSON.\n"
            "- If uncertain, make the closest visually justified estimate."
        )

    def analyse(self, image_path: str) -> ProductSpecs:
        """
        Performs a full vision analysis of the product
        and returns ProductSpecs.
        """
        # read image bytes
        with open(image_path, "rb") as _f:
            image_bytes = _f.read()

        # Compose the complete contents
        response_text = self.model.invoke_with_image(self.system_prompt, image_bytes)

        # Parse as JSON into Pydantic model
        try:
            parsed = ProductSpecs.model_validate_json(response_text)
        except Exception:
            raise ValueError(
                "AnalystAgent: JSON parsing failed. "
                f"Raw output: {response_text}"
            )

        return parsed
