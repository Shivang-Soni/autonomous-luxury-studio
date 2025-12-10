from typing import Any, Dict

from pydantic import BaseModel


class ProductSpecs(BaseModel):
    metal_type: str
    main_stone: Dict[str, Any]
    setting_style: str
    imperfections: str
    misc: str

