# schemas/history_schemas.py
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, constr

class EnhancementHistoryBase(BaseModel):
    # user_id is provided by the system (from the auth token), not the client
    original_image_id: uuid.UUID
    processed_image_id: Optional[uuid.UUID] = None
    parameters_json: constr(max_length=2048)

class EnhancementHistorySchema(EnhancementHistoryBase):
    id: uuid.UUID
    user_id: uuid.UUID # Include user_id in the response
    created_at: datetime

    class Config:
        from_attributes = True