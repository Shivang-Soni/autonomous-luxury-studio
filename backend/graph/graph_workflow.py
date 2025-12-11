from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from schemas import GraphState
from agents.analyst import AnalystAgent
from agents.art_director import DirectorAgent
from agents.judge import JudgeAgent
from agents.producer import ProducerAgent
from config import Configuration

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

    def _node_analyst(self, state: GraphState) -> GraphState:
        analysis = self.analyst_agent.analyse(state.product)
        state.analysis = analysis
        return state

    def _node_director(self, state: GraphState) -> GraphState:
        scene_plan = self.director_agent.create_scene(
            specs=state.product,
            analysis=state.analysis
        )
        state.scene_plan = scene_plan
        return state

    def _node_producer(self, state: GraphState) -> GraphState:
        generation = self.producer_agent.generate_final_candidate(
            product_png_path=state.analysis.get("product_png_path"),
            scene_plan=state.scene_plan,
            feedback=state.judgement.get("feedback") if state.judgement else None
        )
        state.generation = generation
        return state

    def _node_judge(self, state: GraphState) -> GraphState:
        score, feedback = self.judge_agent.evaluate(
            original_image_path=state.analysis.get("product_png_path"),
            candidate_image_path=state.generation.get("generated_image_path")
        )
        state.judgement = {
            "score": score,
            "feedback": feedback
        }
        return state

    def _should_retry(self, state: GraphState) -> str:
        score = state.judgement.get("score", 100)
        if score > self.threshold:
            return "end"
        if state.retries >= self.max_retries:
            return "end"
        state.retries += 1
        state.scene_plan = self.director_agent.correct_scene(
            scene_plan=state.scene_plan,
            feedback=state.judgement.get("feedback")
        )
        return "producer"

    def build(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("analyst", self._node_analyst)
        workflow.add_node("director", self._node_director)
        workflow.add_node("producer", self._node_producer)
        workflow.add_node("judge", self._node_judge)

        workflow.set_entry_point("analyst")

        workflow.add_edge("analyst", "director")
        workflow.add_edge("director", "producer")
        workflow.add_edge("producer", "judge")

        workflow.add_conditional_edge(
            "judge",
            self._should_retry,
            {
                "producer": "producer",
                "end": END
            }
        )

        return workflow.compile(checkpointer=self.checkpointer)
