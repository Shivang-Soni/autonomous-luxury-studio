from typing import Dict, Any

from google import genai

from backend.schemas import ProductSpecs


class AnalystAgent:
    """
    The Gemologist Node
    Extracts ground truth features from product images.
    Ensures accurate downstream generations
    """

    def __init__(self):
        
