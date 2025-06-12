# schemas/image_schemas.py
import uuid
from typing import Optional, List
from pydantic import BaseModel

# Pydantic model for Apply Preset Request
class ApplyPresetRequest(BaseModel):
    image_id: uuid.UUID

# Pydantic models for Image Enhancement
class EnhancementParameters(BaseModel):
    brightness_target: float
    contrast_target: float
    saturation_target: float
    background_blur_radius: int
    crop_rect: List[int]  # [x, y, width, height]
    face_smooth_intensity: float

class AutoEnhancementResponse(BaseModel):
    image_id: uuid.UUID
    enhancement_parameters: EnhancementParameters
    message: Optional[str] = None

# Pydantic models for Manual Image Enhancement Request
class EnhancementRequestParams(BaseModel): # Mirrors EnhancementParameters
    brightness_target: float
    contrast_target: float
    saturation_target: float
    background_blur_radius: int
    crop_rect: List[int]  # [x, y, width, height]
    face_smooth_intensity: float

class ImageEnhancementRequest(BaseModel):
    image_id: uuid.UUID
    parameters: EnhancementRequestParams
