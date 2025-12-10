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
