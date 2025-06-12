import uuid
from typing import List, Optional
from pydantic import BaseModel

class ImageQualityMetrics(BaseModel):
    brightness: float
    contrast: float

# Pydantic model for content moderation result
class ContentModerationResult(BaseModel): # Corrected to BaseModel
    is_approved: bool
    rejection_reason: Optional[str] = None

# Pydantic models for Face Detection API
class FaceBoundingBox(BaseModel):
    box: List[int]  # [x, y, width, height]
    confidence: Optional[float] = None

class FaceDetectionResponse(BaseModel):
    faces: List[FaceBoundingBox]
    image_id: uuid.UUID # Changed from int
    message: Optional[str] = None

class ImageQualityAnalysisResponse(BaseModel):
    image_id: uuid.UUID
    quality_metrics: ImageQualityMetrics
    insights: List[str]
    message: Optional[str] = None
