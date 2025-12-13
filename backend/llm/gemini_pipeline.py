from google import genai
from google.genai import types
from llm.base import BaseLLMClient


class GeminiAdapter(BaseLLMClient):

    def __init__(self, model: str):
        self.client = genai.Client()
        self.model = model

    def invoke(self, prompt: str) -> str:
        res = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return res.text

    def invoke_with_image(self, prompt: str, image_bytes: bytes) -> str:
        part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
        res = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, part]
        )
        return res.text
