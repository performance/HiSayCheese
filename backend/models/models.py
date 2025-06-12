# models/models.py
import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

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

class UserPreset(Base):
    __tablename__ = "user_presets"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    preset_name = Column(String, nullable=False)
    parameters_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class EnhancementHistory(Base):
    __tablename__ = "enhancement_history"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    original_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False, index=True)
    processed_image_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True, index=True)
    parameters_json = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)