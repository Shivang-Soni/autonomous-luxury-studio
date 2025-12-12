from typing import Any, Optional

from llm.gemini_pipeline import GeminiClient
from schemas import ScenePlan, ImageResult
from config import Configuration

config = Configuration()


class ProducerAgent:
    """
    The Producer Node

    Responsibilities:
    - Converts a validated ScenePlan into an actual image generation request.
    - Forwards instructions to Gemini's image generation model (Imagen 3).
    - Returns the generated base64 image along with metadata.
    """

    def __init__(self, model: str = config.IMAGE_MODEL):
        # Model must be Imagen 3 or another image-capable Gemini model
        self.model = GeminiClient(model=model)

        self.system_prompt = (
            "You are the Image Producer. Your job is to take a validated ScenePlan\n"
            "and convert it into a directly runnable image generation instruction.\n\n"
            "You MUST output STRICT JSON with the following structure:\n"
            "{\n"
            '   "prompt": "string",\n'
            '   "negative_prompt": "string",\n'
            '   "width": int,\n'
            '   "height": int,\n'
            '   "infer": boolean\n'
            "}\n\n"
            "RULES:\n"
            "- Width and height must be provided explicitly.\n"
            "- Negative prompt must not be empty.\n"
            "- JSON must contain no commentary.\n"
        )

    def generate_image(self, plan: ScenePlan) -> ImageResult:
        """
        Uses the scene plan to request an image generation response.
        """

        user_message = (
            "Convert the following ScenePlan into STRICT JSON image instructions:\n\n"
            f"{plan.model_dump_json()}"
        )

        prompt = f"{self.system_prompt}\n\n{user_message}"

        # Step 1: Convert ScenePlan → Image instruction JSON
        instruction_raw = self.model.invoke(prompt)

        try:
            instruction = ScenePlan.ImageInstruction.model_validate_json(instruction_raw)
        except Exception as exc:
            raise ValueError(
                "ProducerAgent: Failed to parse image instruction JSON.\n"
                f"Raw:\n{instruction_raw}\n"
                f"Error: {exc}"
            )

        # Step 2: Invoke actual image generation (Imagen 3)
        # GeminiClient.invoke_image returns base64 image
        image_b64 = self.model.invoke_image(
            prompt=instruction.prompt,
            negative_prompt=instruction.negative_prompt,
            width=instruction.width,
            height=instruction.height,
        )

        return ImageResult(
            image_base64=image_b64,
            metadata={"source": "imagen-3", "width": instruction.width, "height": instruction.height},
        )
