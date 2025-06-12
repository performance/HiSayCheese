# schemas/preset_schemas.py
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, constr

MAX_STRING_LENGTH = 255

class PresetBase(BaseModel):
    preset_name: constr(max_length=MAX_STRING_LENGTH)
    parameters_json: constr(max_length=2048)

class PresetCreate(PresetBase):
    pass

class PresetUpdate(PresetBase):
    preset_name: Optional[constr(max_length=MAX_STRING_LENGTH)] = None
    parameters_json: Optional[constr(max_length=2048)] = None

class PresetSchema(PresetBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True