from typing import Tuple, Dict

from llm.gemini_pipeline import GeminiClient
from config import Configuration

config = Configuration()


class JudgeAgent:
    """
    JudgeAgent Node

    Responsiblities:
    - Compare Original to the Generated Image
    - Evaluate critical visual aspects: cut, petals, metal, anatomy, shape
      and so on
    - Return the score (0-100) and a structured feedback
      for the Producer to analyse
    """

    def __init__(self, model: str = config.JUDGE_MODEL):
        self.model = GeminiClient(model=model)

        self.system_prompt = (
            "You are a meticulous Visual QA AI specialized in luxury jewelry photography. "
            "Compare the Original product image and the Generated result.\n\n"
            "Evaluate STRICTLY, following are some examples:\n"
            "- Is the diamond cut identical?\n"
            "- Are there exactly 5 flower petals?\n"
            "- Does the metal look like real metal and not plastic?\n"
            "- Are the model's fingers anatomically correct?\n"
            "Return a STRICT JSON object:\n"
            "{\n"
            '  "score": 0-100,\n'
            '  "feedback": "string description of required corrections"\n'
            "}\n"
            "RULES:\n"
            "- JSON only, no extra commentary.\n"
            "- If uncertain, make the closest visually justified estimate."
        )

    def evaluate(
            self,
            original_image_path: str,
            candidate_image_path: str,
        ) -> Tuple[int, Dict]:
        """
        Compares the original and the candidate images STRICTLY.
        Returns score in the range from 0 to 100,
        as well as a feedback dictionary.
        """

        # Load images
        original_part = self.model.load_image(original_image_path)
        candidate_part = self.model.load_image(candidate_image_path)

        # Orchestration of the individual components
        contents = [
            self.system_prompt,
            original_part,
            candidate_part
            ]
        
        raw_output = self.model.invoke_with_image(contents)

        # Parse output
        try:
            parsed = eval(raw_output)
            score = int(parsed.get("score", "100"))
            feedback = parsed.get("feedback", "")
            feedback_dict = {
                "score": score,
                "feedback": feedback
            }
        except Exception:
            score = 100
            feedback_dict = {
                "score": score,
                "feedback": "Unable to parse Q/A output."
            }
        
        return score, feedback_dict
