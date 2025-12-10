from typing import Callable

from backend.schemas import GraphState, ScenePlan
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

        self.threshold = config.MIN_ACCEPTED_SCORE
        self.max_retries = config.MAX_RETRIES
    
    def run(
            self,
            product_specs,
            image_path
    ) -> GraphState:
        """
        Main entrypoint.
        Accepts the ProductSpecs and returns the final GraphState.
        """
        state = GraphState(product=product_specs)
        # Node A: AnalystAgent
        analysis = self.analyst_agent.analyse(product_specs)
        state.analysis = analysis
        # Node B: DirectorAgent
        scene_plan: ScenePlan = self.director_agent.create_scene(analysis)
        state.scene_plan = scene_plan
        # Producer + Agent Loop (C<=>D)
        retries = 0
        last_score = 0
        feedback = None

        while retries <= self.max_retries:
            # Node C: ProducerAgent
            candidate_path = self.producer_agent.generate_final_candidate(
                product_png_path=config.INPUT_DIR,
                scene_plan=state.scene_plan,
                feedback=feedback
                )
            
            state.generation.append(candidate_path)

            # Node D: JudgeAgent
            score, judge_feedback = self.judge_agent.evaluate(
                original_image_path=image_path,
                candidate_image_path=candidate_path
            )

            state_scores.append(score)