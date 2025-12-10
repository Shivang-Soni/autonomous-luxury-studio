from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.schemas import GraphState
from backend.agents.analyst import AnalystAgent
from backend.agents.art_director import DirectorAgent
from backend.agents.judge import JudgeAgent
from backend.agents.producer import ProducerAgent
from backend.config import Configuration

config = Configuration()


class GraphWorkflow:
    """
    Core Workflow Orchestrator
    Executes our Autonomous Luxury Studio pipeline:
    - AnalystAgent: Analyses the product
    - DirectorAgent: Formulates the prompt suitable to the product details
    - ProducerAgent: Generates outputs based on the instructions
      received from the previous agents, as well as based on feedback
      from the JudgeAgent
    - JudgeAgent: evaluates and provides feedback
    - Retries (ReAct Agentic pipeline),
      until the score is deemed to be satisfactory
    """

    def __init__(
            self,
            analyst_agent: AnalystAgent,
            director_agent: DirectorAgent,
            producer_agent: ProducerAgent,
            judge_agent: JudgeAgent
    ):
        self.analyst_agent = analyst_agent
        self.director_agent = director_agent
        self.producer_agent = producer_agent
        self.judge_agent = judge_agent
        self.checkpointer = MemorySaver()

        self.threshold = config.MIN_ACCEPTED_SCORE
        self.max_retries = config.MAX_RETRIES
    
   