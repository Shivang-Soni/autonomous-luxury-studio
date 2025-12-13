from abc import ABC, abstractmethod


class BaseLLMClient(ABC):

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Text prompt -> text response
        """
        pass

    @abstractmethod
    def invoke_with_image(self, prompt: str, image_bytes: bytes) -> str:
        """
        Image + Text -> Text Response
        """
        pass
