import os
from pathlib import Path
import asyncio
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException


from agents.analyst import AnalystAgent
from agents.art_director import DirectorAgent
from agents.judge import JudgeAgent
from agents.producer import ProducerAgent
from graph.graph_workflow import GraphWorkflow
from schemas import GraphState, ProductSpecs
from config import Configuration

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

@router.post("/process/upload")
async def process_single_upload(file: UploadFile = File(...)):
    try:
        contents = file.read()
        state = GraphState()
        result = await workflow_builder.run
