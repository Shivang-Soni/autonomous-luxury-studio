from typing import Any, Dict, Optional

from pydantic import BaseModel


class MainStone(BaseModel):
    cut: str
    color: str
    clarity: str
    carat: Optional[str] = None


class ProductSpecs(BaseModel):
    metal_type: str
    main_stone: MainStone
    setting_style: str
    unique_imperfections: str


class LightingMap(BaseModel):
    source_direction: str
    temperature: str


class ScenePlan(BaseModel):
    prompt: str
    negative_prompt: str
    lighting_map: LightingMap
    inpaint_coordinates: list
