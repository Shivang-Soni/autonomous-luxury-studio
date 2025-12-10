import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.agents.analyst import AnalystAgent
from backend.agents.art_director import DirectorAgent
from backend.agents.judge import JudgeAgent
from backend.agents.producer import ProducerAgent
from backend.graph.graph_workflow import GraphWorkflow
from backend.schemas import GraphState, ProductSpecs
from backend.config import Configuration

router = APIRouter()
config = Configuration()

# Agent Initialisation
analyst = AnalystAgent()
director = DirectorAgent()
producer = ProducerAgent()
judge = JudgeAgent()

workflow_builder = GraphWorkflow(
    analyst,
    director,
    producer,
    judge
)

