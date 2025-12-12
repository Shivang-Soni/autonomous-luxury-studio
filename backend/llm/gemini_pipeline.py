from google import genai
from google.genai import types


class GeminiClient:
    """
    Gemini LLM Wrapper client
    """

    def __init__(self, model: str = "gemini-2.0-pro-exp-02-05"):
        # Client folgt der aktuellen Google-Dokumentation
        self.client = genai.Client()
        self.model = model

    def invoke(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    def invoke_with_image(self, prompt: str, image_bytes: bytes) -> str:
        """
        Vision + Text pipeline, konform zur aktuellen Google-API.
        """
        part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, part],
        )
        return response.text
