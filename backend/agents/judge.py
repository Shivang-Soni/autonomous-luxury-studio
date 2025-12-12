from typing import Any, Dict

from llm.gemini_pipeline import GeminiClient
from schemas import ScenePlan, ProductSpecs, JudgeEvaluation
from config import Configuration

config = Configuration()


class JudgeAgent:
    """
    The Judge Node

    Responsibilities:
    - Evaluates whether the generated scene plan is:
        • Accurate to the product specs
        • Brand-aligned
        • Physically realistic
        • Cinematically coherent
    - Returns strict JSON evaluation to ensure downstream consistency.
    """

    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = GeminiClient(model=model)

        self.system_prompt = (
            "You are the Senior Creative Judge for 64 Facets.\n"
            "You evaluate the realism, accuracy, and brand validity of jewelry scene plans.\n\n"
            "You MUST output a STRICT JSON object with the following structure:\n"
            "{\n"
            '   "score": float,  // 0.0 - 10.0\n'
            '   "is_approved": boolean,\n'
            '   "issues": [ "string", ... ],\n'
            '   "recommendations": [ "string", ... ]\n'
            "}\n\n"
            "EVALUATION RULES:\n"
            "- Score must reflect luxury brand standards.\n"
            "- Approve only if scene plan is feasible and editorial-grade.\n"
            "- Be strict: unrealistic lighting, impossible geometry, or noisy prompts must be penalized.\n"
            "- JSON must be VALID and contain ZERO commentary.\n"
        )

    def evaluate(self, specs: ProductSpecs, plan: ScenePlan) -> JudgeEvaluation:
        """
        Evaluates a scene plan against product specs using the Gemini model.
        """

        user_message = (
            "Evaluate the following object pair.\n\n"
            "Product Specifications:\n"
            f"{specs.model_dump_json()}\n\n"
            "Scene Plan:\n"
            f"{plan.model_dump_json()}\n\n"
            "Return STRICT evaluation JSON."
        )

        prompt = f"{self.system_prompt}\n\n{user_message}"

        raw_output = self.model.invoke(prompt)

        try:
            evaluation = JudgeEvaluation.model_validate_json(raw_output)
        except Exception as exc:
            raise ValueError(
                "JudgeAgent: JSON decoding failed.\n"
                f"Raw model output:\n{raw_output}\n"
                f"Validation error: {exc}"
            )

        return evaluation
