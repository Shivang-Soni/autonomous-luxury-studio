from typing import Union, List

from google import genai
from google.genai import types


class GeminiClient:
    """
    Gemini LLM Wrapper client, which: 
    - supports text only prompts
    - supports text + image prompts
    """
    def __init__(self, model: str = "gemini-3-pro-preview"):
        self.client = genai.Client()
        self.model = model

    def invoke_text(
            self,
            prompt,
            max_tokens: int = 2048,
            temperature: float = 0.2
            ) -> str:
        """
        Generates from simple text prompt.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
                )
            )

        return response.text

    def invoke_with_image(
            self,
            contents: Union[str, List[Union[str, types.Part]]],
            max_tokens: int = 2048,
            temperature: float = 0.2
            ) -> str:
        """
        Unified text + image generation
        """

        payload = []

        for c in contents:
            if isinstance(c, str):
                payload.append(c)
            elif isinstance(c, types.Part):
                payload.append(c)
            else:
                raise ValueError(f"Unsupported content type: {type(c)}")

        response = self.client.models.generate_content(
            model=self.model,
            contents=payload,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
        )

        return response.text

    @staticmethod
    def load_image(path: str, mime_type: str = "image/png") -> types.Part:
        """
        Loads images as types.Part for Vision models
        """
        with open(path, "rb") as f:
            data = f.read()

        return types.Part.from_bytes(
            data=data,
            mime_type=mime_type
        )
