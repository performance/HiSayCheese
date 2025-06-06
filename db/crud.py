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
from models.models import Number, Image, ImageCreate, ImageSchema, User, UserCreate, EnhancementHistory, EnhancementHistoryBase
from typing import Optional, List # Added for Optional type hint
from auth_utils import hash_password

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
def create_image(
    db: Session,
    image: ImageCreate,
    width: Optional[int] = None,
    height: Optional[int] = None,
    format: Optional[str] = None,
    exif_orientation: Optional[int] = None,
    color_profile: Optional[str] = None
) -> models.Image:
    db_image = models.Image(
        filename=image.filename,
        filepath=image.filepath,
        filesize=image.filesize,
        mimetype=image.mimetype,
        width=width,
        height=height,
        format=format,
        exif_orientation=exif_orientation,
        color_profile=color_profile
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def get_image(db: Session, image_id: uuid.UUID) -> models.Image | None:
    return db.query(models.Image).filter(models.Image.id == image_id).first()

# Functions for User model
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate) -> User:
    hashed_pass = hash_password(user.password)
    db_user = User(email=user.email, hashed_password=hashed_pass)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


# Functions for EnhancementHistory model
def create_enhancement_history(db: Session, history_data: EnhancementHistoryBase, user_id: uuid.UUID) -> EnhancementHistory:
    db_history = EnhancementHistory(
        user_id=user_id,
        original_image_id=history_data.original_image_id,
        processed_image_id=history_data.processed_image_id,
        parameters_json=history_data.parameters_json
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history

def get_enhancement_history_by_user(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 10) -> List[EnhancementHistory]:
    return db.query(EnhancementHistory).filter(EnhancementHistory.user_id == user_id).order_by(EnhancementHistory.created_at.desc()).offset(skip).limit(limit).all()
