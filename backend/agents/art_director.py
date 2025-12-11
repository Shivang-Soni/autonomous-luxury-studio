from typing import Dict, Any

from llm.gemini_pipeline import GeminiClient
from schemas import ProductSpecs, ScenePlan
from config import Configuration

config = Configuration()


class DirectorAgent:
    """
    The Art Dirctor Node

    Responsiblities:
    - Produces
        - generation prompts
        - negative prompts
        - lighting schema
        - precise inpainting coordiates
    - Ensures no additional jewellery is introduced
    - Converts ground truth ProductSpecs into
      a cinematic, brand aligned visual narrative.
    """

    def __init__(self, model: str = config.ART_DIRECTOR_MODEL):
        self.model = GeminiClient(model=model)

        self.system_prompt = (
            "You are the Senior Art Director for 64 Facets,"
            " a luxury jewelry studio. "
            "You transform structured gemstone product specifications into a "
            "high-end cinematic visual narrative.\n\n"
            "You MUST output a STRICT JSON object"
            " following exactly this schema:\n"
            "{\n"
            '  "prompt": "string",\n'
            '  "negative_prompt": "string",\n'
            '  "lighting_map": {\n'
            '       "source_direction": "string",\n'
            '       "temperature": "string"\n'
            "   },\n"
            '  "inpaint_coordinates": [x1, y1, x2, y2]\n'
            "}\n\n"
            "BRAND RULES:\n"
            "- Show ONLY the provided jewelry. Never add extra pieces.\n"
            "- Maintain luxury-class, editorial photographic style.\n"
            "- Avoid distortion, extra fingers, reflections of extra jewelry.\n"
            "- Coordinates must correspond to a realistic anatomical placement.\n"
            "- JSON must be valid, minimal, and contain NO extra commentary.\n"
        )

    def create_scene(self, specs: ProductSpecs) -> ScenePlan:
        """
        Converts ProductSpecs to a ScenePlan
        """

        user_message = (
            "Convert the following product specs"
            " STRICTLY into a scene plan JSON:\n\n"
            f"{specs.model_dump_json()}"
        )

        contents = [
            self.system_prompt,
            user_message
        ]

        raw_output = self.model.invoke_text(contents)

        try:
            scene_plan = ScenePlan.model_validate_json(raw_output)
        except Exception as e:
            raise ValueError(
                f"DirectorAgent: JSON Parsing failed."
                f"Raw model output: \n{raw_output}\n"
                f"Validation error: {e}"
            )

        return scene_plan
