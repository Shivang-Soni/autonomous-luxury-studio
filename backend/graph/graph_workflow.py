from typing import Optional

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
    Orchestrates the full 64 Facets pipeline as a directed graph:
    Analyst -> Director -> Producer -> Judge (feedback loop)
    """

    def __init__(
        self,
        analyst: AnalystAgent,
        director: DirectorAgent,
        producer: ProducerAgent,
        judge: JudgeAgent,
    ):
        self.analyst = analyst
        self.director = director
        self.producer = producer
        self.judge = judge

        self.checkpointer = MemorySaver()
        self.threshold = config.MIN_ACCEPTED_SCORE
        self.max_retries = config.MAX_RETRIES

    def _node_analyst(self, state: GraphState) -> GraphState:
        state.analysis = self.analyst.analyse(state.product)
        return state

    def _node_director(self, state: GraphState) -> GraphState:
        state.scene_plan = self.director.create_scene(state.analysis)
        return state

    def _node_producer(self, state: GraphState) -> GraphState:
        state.generation = self.producer.generate_final_candidate(
            product_png_path=state.analysis.product_png_path,
            scene_plan=state.scene_plan,
            feedback=state.judgement.feedback if state.judgement else None,
        )
        return state

    def _node_judge(self, state: GraphState) -> GraphState:
        score, feedback = self.judge.evaluate(
            original_image_path=state.analysis.product_png_path,
            candidate_image_path=state.generation.generated_image_path,
        )
        state.judgement = {"score": score, "feedback": feedback}
        return state

    def _should_retry(self, state: GraphState) -> str:
        score = state.judgement.get("score", 100)
        if score >= self.threshold or state.retries >= self.max_retries:
            return "end"

        state.retries += 1
        state.scene_plan = self.director.correct_scene(
            scene_plan=state.scene_plan, feedback=state.judgement.get("feedback")
        )
        return "producer"

    def build(self) -> "GraphWorkflow":
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
            {"producer": "producer", "end": END},
        )

        self.workflow = workflow.compile(checkpointer=self.checkpointer)
        return self

    def invoke(self, state: GraphState) -> GraphState:
        return self.workflow.run(state)
