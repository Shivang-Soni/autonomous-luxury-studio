import os
import json
from pathlib import Path
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

INPUT_DIR = Path(config.INPUT_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analyses the product Image it have diamonds,metal and all.
analyst = AnalystAgent()

# Director Agent decides the background of the generated image.
director = DirectorAgent()

# Placement of the Jwellery and generation of the Result.
producer = ProducerAgent()

#Analyses the image and give the Score and provide the Feedback Working as React Agent.
judge = JudgeAgent()

workflow = GraphWorkflow(analyst, director, producer, judge).build()


def clear_directory(directory: Path):
    for f in directory.iterdir():
        f.unlink(missing_ok=True)


def run_single(image_path: Path) -> GraphState:
    specs = ProductSpecs(image_path=str(image_path))
    state = GraphState(product=specs)
    state = workflow.invoke(state)

    if state.generation:
        generated_bytes = state.generation
        img_path = OUTPUT_DIR / f"{image_path.stem}_generated.png"
        with open(img_path, "wb") as f:
            f.write(generated_bytes)

    return state


def save_result(image_path: Path, state: GraphState):
    out_path = OUTPUT_DIR / f"{image_path.stem}_result.json"
    payload = {
        "image": str(image_path),
        "analysis": state.analysis,
        "scene_plan": state.scene_plan.model_dump() if state.scene_plan else None,
        "generation_file": str(OUTPUT_DIR / f"{image_path.stem}_generated.png") if state.generation else None,
        "judgement": state.judgement,
        "retries": state.retries
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)


def batch_process_folder():
    files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    if not files:
        raise HTTPException(status_code=400, detail="No valid images in INPUT_DIR.")

    results = []

    for img in files:
        try:
            state = run_single(img)
            save_result(img, state)
            results.append({"file": img.name, "status": "processed"})
        except Exception as e:
            results.append({"file": img.name, "error": str(e)})

    clear_directory(INPUT_DIR)
    return results


@router.post("/process/upload-batch")
async def upload_and_process_batch(files: List[UploadFile] = File(...)):
    for file in files:
        save_path = INPUT_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(await file.read())

    results = batch_process_folder()
    return {
        "input_count": len(files),
        "output_dir": str(OUTPUT_DIR),
        "results": results
    }


@router.post("/process/folder")
async def process_existing_folder():
    results = batch_process_folder()
    return {
        "output_dir": str(OUTPUT_DIR),
        "results": results
    }
