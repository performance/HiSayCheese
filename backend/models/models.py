import uuid
from datetime import datetime
from typing import Optional
import re

from sqlalchemy import Column, DateTime, Integer, String, ForeignKey, Index, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr, conint, constr, validator

Base = declarative_base()

# Constants for validation
MAX_STRING_LENGTH = 255
MAX_FILE_SIZE_MB = 100  # Example: 100MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_PASSWORD_LENGTH = 8

class Number(Base):
    __tablename__ = "numbers"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(Integer)

class Image(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String)
    filepath = Column(String)
    filesize = Column(Integer)
    mimetype = Column(String)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String, nullable=True)
    exif_orientation = Column(Integer, nullable=True)
    color_profile = Column(String, nullable=True)
    rejection_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String, nullable=True, index=True, unique=True)
    verification_token_expires_at = Column(DateTime, nullable=True)


# Pydantic schemas for User
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: constr(min_length=MIN_PASSWORD_LENGTH)

    @validator('password')
    def password_strength(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain an uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain a lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain a digit')
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError('Password must contain a special character')
        return v

class UserSchema(UserBase):
    id: uuid.UUID
    created_at: datetime
    is_verified: bool # Added is_verified field

    class Config:
        orm_mode = True


class UserPreset(Base):
    __tablename__ = "user_presets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    preset_name = Column(String, nullable=False)
    parameters_json = Column(String, nullable=False) # To store enhancement parameters as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Optional: Add a unique constraint for user_id and preset_name if needed
    # __table_args__ = (UniqueConstraint('user_id', 'preset_name', name='uq_user_preset_name'),)


# Pydantic schemas for UserPreset
class PresetBase(BaseModel):
    preset_name: constr(max_length=MAX_STRING_LENGTH)
    parameters_json: constr(max_length=2048) # Assuming JSON strings can be longer

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
        orm_mode = True


class EnhancementHistory(Base):
    __tablename__ = "enhancement_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    original_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, index=True)
    processed_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True, index=True)
    parameters_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic schemas for EnhancementHistory
class EnhancementHistoryBase(BaseModel):
    user_id: uuid.UUID
    original_image_id: uuid.UUID
    processed_image_id: Optional[uuid.UUID] = None
    parameters_json: constr(max_length=2048) # Assuming JSON strings can be longer

class EnhancementHistorySchema(EnhancementHistoryBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True


class NumberBase(BaseModel):
    value: conint(ge=0) # Example: ensure value is non-negative

class NumberCreate(NumberBase):
    pass

class NumberSchema(NumberBase):
    id: int

    class Config:
        orm_mode = True

# Pydantic schema for creating an Image
class ImageCreate(BaseModel):
    filename: constr(max_length=MAX_STRING_LENGTH)
    filepath: Optional[constr(max_length=MAX_STRING_LENGTH * 2)] = None # Filepaths can be longer
    filesize: conint(gt=0, le=MAX_FILE_SIZE_BYTES)
    mimetype: constr(max_length=MAX_STRING_LENGTH)
    width: Optional[conint(gt=0)] = None
    height: Optional[conint(gt=0)] = None
    format: Optional[constr(max_length=50)] = None
    exif_orientation: Optional[conint(ge=1, le=8)] = None # EXIF orientation is typically 1-8
    color_profile: Optional[constr(max_length=MAX_STRING_LENGTH)] = None
    rejection_reason: Optional[constr(max_length=MAX_STRING_LENGTH * 2)] = None # Rejection reasons can be longer

# Pydantic schema for reading/returning an Image
class ImageSchema(ImageCreate): # Inherits fields from ImageCreate
    id: uuid.UUID
    created_at: datetime
    presigned_url: Optional[str] = None # Added new field for presigned URL
    # width, height, format, etc. inherit constraints from ImageCreate via inheritance

    class Config:
        orm_mode = True
