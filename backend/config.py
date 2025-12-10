import os

from dotenv import load_dotenv

load_dotenv()


class Configuration:
    def __init__(self):
        self.ANALYST_MODEL = os.getenv("ANALYST_MODEL", "")
        self.ART_DIRECTOR_MODEL = os.getenv("ART_DIRECTOR_MODEL", "")
        self.JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
        self.PRODUCER_MODEL = os.getenv("PRODUCER_MODEL", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
