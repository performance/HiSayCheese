import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel

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
    created_at = Column(DateTime, default=datetime.utcnow)

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
    filepath: str
    filesize: int
    mimetype: str

# Pydantic schema for reading/returning an Image
class ImageSchema(ImageCreate): # Inherits fields from ImageCreate
    id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True
