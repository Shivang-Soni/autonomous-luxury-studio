from typing import Optional, Dict
from pathlib import Path
import uuid

from schemas import ScenePlan
from llm.gemini_pipeline import GeminiClient
from config import Configuration

config = Configuration()


class ProducerAgent:
    """
    Producer Node

    Responsibilities:
    - Generate base scene without the jewellery.
    - Inpaint original product PNG into the generated scene.
    - Abide by ScenePlan coordinates, lighting and negative prompts.
    - Accept feedback from the Judge and attempt up to max_retries.
    """

    def __init__(self, model: str = config.PRODUCER_MODEL, max_retries: int = 3):
        self.client = GeminiClient(model=model)
        self._max_retries = max_retries

    def generate_scene_base(self, scene_plan: ScenePlan) -> bytes:
        """
        Generate base scene without jewellery.
        Returns image as bytes.
        """

        prompt = (
            f"{scene_plan.prompt}\n\n"
            "DO NOT include the jewellery yet. "
            "ONLY generate the environment, model, background, and lighting. "
            f"Follow the lighting_map: source={scene_plan.lighting_map.source_direction}, "
            f"temperature={scene_plan.lighting_map.temperature}. "
            f"Negative prompt: {scene_plan.negative_prompt}"
        )

        response = self.client.invoke_text(prompt)
        return response.encode("utf-8")

    def inpaint_product(
        self,
        base_scene_bytes: bytes,
        product_png_path: str,
        scene_plan: ScenePlan
    ) -> bytes:
        """
        Inpaints the original product PNG into the base generated scene.
        """
        product_part = self.client.load_image(path=product_png_path)

        prompt = (
            "Inpaint the jewellery at the following coordinates: "
            f"{scene_plan.inpaint_coordinates}. "
            "Ensure shadows, lighting, and skin interactions are photorealistic."
        )

        contents = [
            base_scene_bytes,
            product_part,
            prompt
        ]

        response_text = self.client.invoke_with_image(contents)
        return response_text.encode("utf-8")

    def generate_final_candidate(
        self,
        product_png_path: str,
        scene_plan: ScenePlan,
        feedback: Optional[Dict] = None,
    ) -> bytes:
        """
        End-to-end ProducerAgent pipeline.
        """

        candidate_bytes = None

        for attempt in range(self._max_retries):

            base_scene = self.generate_scene_base(scene_plan)

            candidate_bytes = self.inpaint_product(
                base_scene_bytes=base_scene,
                product_png_path=product_png_path,
                scene_plan=scene_plan
            )

            if feedback:
                score = feedback.get("score", 100)
                if score >= 90:
                    break

                scene_feedback = feedback.get("feedback", "")
                scene_plan.prompt += (
                    f" Adjust the scene based on the following feedback: {scene_feedback}"
                )

        return candidate_bytes

    def save_candidates(self, candidate_bytes: bytes, output_dir: str) -> str:
        """
        Saves the finalised candidate image in the output directory
        and returns the file path.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_dir) / f"candidate_{uuid.uuid4().hex}.png"
        with open(file_path, "wb") as file:
            file.write(candidate_bytes)

        return str(file_path)
