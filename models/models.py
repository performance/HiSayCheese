import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Integer, String
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
