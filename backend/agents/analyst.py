from typing import Dict, Any

from google import genai

from schemas import ProductSpecs
from llm.gemini_pipeline import GeminiClient
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
            "Analyze the provided product image "
            "and produce a STRICT JSON object "
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

        img_part = self.model.load_image(image_path)

        # Compose the complete contents list
        contents = [
            img_part,
            self.system_prompt
        ]

        response_text = self.model.invoke_with_image(contents)

        # Parse as JSON
        try:
            parsed_json = ProductSpecs.model_validate_json(response_text)
        except Exception:
            raise ValueError(
                f"AnalystAgent: JSON parsing failed."
                f"Raw output: {response_text}"
                )

        return parsed_json
