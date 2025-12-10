from google import genai


class GeminiClient:
    """
    Gemini LLM Wrapper client
    """
    def __init__(self, model: str = "gemini-3-pro-preview"):
        self.client = genai.Client()
        self.model = model

    def invoke(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text
