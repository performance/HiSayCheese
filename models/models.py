import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr

Base = declarative_base()

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


# Pydantic schemas for User
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str # min_length=8 will be handled by endpoint logic

class UserSchema(UserBase):
    id: uuid.UUID
    created_at: datetime

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
    preset_name: str
    parameters_json: str

class PresetCreate(PresetBase):
    pass

class PresetUpdate(PresetBase):
    preset_name: Optional[str] = None
    parameters_json: Optional[str] = None

class PresetSchema(PresetBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True


class EnhancementHistory(Base):
    __tablename__ = "enhancement_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    processed_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True)
    parameters_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic schemas for EnhancementHistory
class EnhancementHistoryBase(BaseModel):
    user_id: uuid.UUID
    original_image_id: uuid.UUID
    processed_image_id: Optional[uuid.UUID] = None
    parameters_json: str

class EnhancementHistorySchema(EnhancementHistoryBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True


class NumberBase(BaseModel):
    value: int

class NumberCreate(NumberBase):
    pass

class NumberSchema(NumberBase):
    id: int

    class Config:
        orm_mode = True

# Pydantic schema for creating an Image
class ImageCreate(BaseModel):
    filename: str
    filepath: Optional[str] = None
    filesize: int
    mimetype: str
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    exif_orientation: Optional[int] = None
    color_profile: Optional[str] = None
    rejection_reason: Optional[str] = None

# Pydantic schema for reading/returning an Image
class ImageSchema(ImageCreate): # Inherits fields from ImageCreate
    id: uuid.UUID
    created_at: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    exif_orientation: Optional[int] = None
    color_profile: Optional[str] = None
    rejection_reason: Optional[str] = None

    class Config:
        orm_mode = True
