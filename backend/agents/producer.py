from typing import Optional, Dict, Any
from pathlib import Path
import uuid

from google import genai
from schemas import ScenePlan
from llm.gemini_pipeline import GeminiClient
from config import Configuration

config = Configuration()


class ProducerAgent:
    """
    Producer Node

    Responsiblities:
    - Generate full scene without jewellery
    - Inpaint original product PNG into generated scene
    - Abide by ScenePlan coordiantes, lighting and negative prompts
    - Accept feedback from the Judge and attempt upto max_retries
    """

    def __init__(self, model: str = config.PRODUCER_MODEL, max_retries: int = 3):
        self.client = GeminiClient(model=model)
        self._max_retries = max_retries

    def generate_scene_base(self, scene_plan: ScenePlan) -> bytes:
        """
        Generate base scene without jewellery.
        Returns the image as bytes.
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

        # Create inpainting prompt
        prompt = (
            f"Inpaint the jewellery at the following coordinates:"
            f"{scene_plan.inpaint_coordinates}."
            f"\nEnsure shadows, lighting, and skin interactions are photorealstic."

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
        End to end ProducerAgent pipeline:
        Base Scene -> Inpainting -> Feedback Loop
        """

        attempt = 0
        candidate_bytes = None

        while attempt<self._max_retries:
            attempt += 1
            # Generate base scene
            base_scene = self.generate_scene_base(scene_plan)
            # Inpaint Original PNG onto the generated base scene
            candidate_bytes = self.inpaint_product(
                base_scene_bytes=base_scene,
                product_png_path=product_png_path,
                scene_plan=scene_plan
                )
            # Check feedback from the JudgeAgent, if present
            if feedback:
                if feedback.get("score", 100) >= 90:
                    break
            else:
                # Adjust output based on the feedback
                scene_plan.prompt += f"Adjust based on the feedback from the JudgeAgent: {feedback.get("feedback", "")}"

        return candidate_bytes
    
    def save_candidates(self, candidate_bytes: bytes, output_dir: str) -> str:
        """
        Saves the finalised candidate image in the output directory
        and returns the file path.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(output_dir)/f"candidate_{uuid.uuid4().hex}.png"
        with open(file_path, "wb") as file:
            file.write(candidate_bytes)
        return str(file_path)
