import uuid # Added import for uuid
from sqlalchemy.orm import Session
from models import models # Updated import to access Image model and Pydantic schemas
# from .database import Number # This line is problematic as Number is now in models.models

# Assuming Number model is now also from models.models
# from models.models import Number, Image # More specific imports
# from models.models import NumberCreate, NumberSchema # For consistency if we define Pydantic schemas for Number in models.py
# For now, let's assume the existing Number functions work with Number from models.models if it's defined there
# or if .database still correctly provides it (which it does after previous subtask)

# Let's adjust imports for clarity and correctness based on previous steps
from models.models import Number, Image, ImageCreate, ImageSchema

def get_number(db: Session):
    return db.query(models.Number).first() # Adjusted to use models.Number

def create_or_update_number(db: Session, value: int):
    db_number = get_number(db) # This uses the updated get_number
    if db_number:
        db_number.value = value
    else:
        db_number = models.Number(value=value) # Adjusted to use models.Number
        db.add(db_number)
    db.commit()
    db.refresh(db_number)
    return db_number

def increment_number(db: Session):
    db_number = get_number(db)
    if db_number:
        db_number.value += 1
        db.commit()
        db.refresh(db_number)
        return db_number
    return None

# Functions for Image model
def create_image(db: Session, image: ImageCreate) -> models.Image:
    db_image = models.Image(
        filename=image.filename,
        filepath=image.filepath,
        filesize=image.filesize,
        mimetype=image.mimetype
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def get_image(db: Session, image_id: uuid.UUID) -> models.Image | None:
    return db.query(models.Image).filter(models.Image.id == image_id).first()
