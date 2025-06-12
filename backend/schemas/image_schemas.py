# schemas/image_schemas.py
import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, constr, conint

# Constants can be defined here or imported from a central config
MAX_STRING_LENGTH = 255
MAX_FILE_SIZE_MB = 15  # Keep this consistent with router's constant
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Pydantic schema for creating an Image
class ImageCreate(BaseModel):
    filename: constr(max_length=MAX_STRING_LENGTH)
    filepath: Optional[constr(max_length=MAX_STRING_LENGTH * 2)] = None
    filesize: conint(gt=0, le=MAX_FILE_SIZE_BYTES)
    mimetype: constr(max_length=MAX_STRING_LENGTH)
    width: Optional[conint(gt=0)] = None
    height: Optional[conint(gt=0)] = None
    format: Optional[constr(max_length=50)] = None
    exif_orientation: Optional[conint(ge=1, le=8)] = None
    color_profile: Optional[constr(max_length=MAX_STRING_LENGTH)] = None
    rejection_reason: Optional[constr(max_length=MAX_STRING_LENGTH * 2)] = None

# Pydantic schema for reading/returning an Image
class ImageSchema(ImageCreate): # Inherits fields from ImageCreate
    id: uuid.UUID
    created_at: datetime
    presigned_url: Optional[str] = None # For returning a temporary URL to the client

    class Config:
        from_attributes = True # Changed from orm_mode for Pydantic v2+

# You can move other image-related schemas here as well
class ProcessedImageResponse(BaseModel):
    original_image_id: uuid.UUID
    processed_image_id: Optional[uuid.UUID] = None
    processed_image_path: Optional[str] = None
    message: str
    error: Optional[str] = None