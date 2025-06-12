# db/crud.py

import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy.orm import Session

# --- CORRECTED IMPORTS ---
# Import only SQLAlchemy models from the 'models' module.
from models import models as db_models

# Import Pydantic schemas from the 'schemas' module.
# This clearly separates API data shapes from DB table structures.
from schemas import (
    image_schemas,
    user_schemas,
    preset_schemas,
    history_schemas,
    number_schemas
)

from auth_utils import hash_password

# --- Number CRUD ---
def get_number(db: Session) -> Optional[db_models.Number]:
    return db.query(db_models.Number).first()

def create_or_update_number(db: Session, number_in: number_schemas.NumberCreate) -> db_models.Number:
    db_number = get_number(db)
    if db_number:
        db_number.value = number_in.value
    else:
        db_number = db_models.Number(value=number_in.value)
        db.add(db_number)
    db.commit()
    db.refresh(db_number)
    return db_number

def increment_number(db: Session) -> Optional[db_models.Number]:
    db_number = get_number(db)
    if db_number:
        db_number.value += 1
        db.commit()
        db.refresh(db_number)
    return db_number

# --- Image CRUD ---
# SIMPLIFIED: The function now takes a single Pydantic object.
# The router/service layer is responsible for populating this object correctly.
def create_image(db: Session, image: image_schemas.ImageCreate) -> db_models.Image:
    """Creates a new image record in the database from a Pydantic schema."""
    # model_dump() converts the Pydantic model to a dictionary,
    # which can be used to instantiate the SQLAlchemy model.
    image_data = image.model_dump()
    db_image = db_models.Image(**image_data)
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def get_image(db: Session, image_id: uuid.UUID) -> Optional[db_models.Image]:
    return db.query(db_models.Image).filter(db_models.Image.id == image_id).first()

# --- User CRUD ---
def get_user_by_email(db: Session, email: str) -> Optional[db_models.User]:
    return db.query(db_models.User).filter(db_models.User.email == email).first()

def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[db_models.User]:
    return db.query(db_models.User).filter(db_models.User.id == user_id).first()

def create_user(db: Session, user: user_schemas.UserCreate) -> db_models.User:
    hashed_pass = hash_password(user.password)
    verification_token = secrets.token_urlsafe(32)
    token_expiry_time = datetime.utcnow() + timedelta(hours=24)

    db_user = db_models.User(
        email=user.email,
        hashed_password=hashed_pass,
        is_verified=False,
        verification_token=verification_token,
        verification_token_expires_at=token_expiry_time
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_verification_token(db: Session, token: str) -> Optional[db_models.User]:
    return db.query(db_models.User).filter(db_models.User.verification_token == token).first()

def mark_user_as_verified(db: Session, user: db_models.User) -> db_models.User:
    user.is_verified = True
    user.verification_token = None
    user.verification_token_expires_at = None
    db.commit()
    db.refresh(user)
    return user

# --- EnhancementHistory CRUD ---
def create_enhancement_history(db: Session, history_data: history_schemas.EnhancementHistoryBase, user_id: uuid.UUID) -> db_models.EnhancementHistory:
    db_history = db_models.EnhancementHistory(
        user_id=user_id,
        original_image_id=history_data.original_image_id,
        processed_image_id=history_data.processed_image_id,
        parameters_json=history_data.parameters_json
    )
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history

def get_enhancement_history_by_user(db: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 10) -> List[db_models.EnhancementHistory]:
    return db.query(db_models.EnhancementHistory).filter(db_models.EnhancementHistory.user_id == user_id).order_by(db_models.EnhancementHistory.created_at.desc()).offset(skip).limit(limit).all()

# --- UserPreset CRUD ---
def create_user_preset(db: Session, preset_data: preset_schemas.PresetCreate, user_id: uuid.UUID) -> db_models.UserPreset:
    db_preset = db_models.UserPreset(
        user_id=user_id,
        preset_name=preset_data.preset_name,
        parameters_json=preset_data.parameters_json
    )
    db.add(db_preset)
    db.commit()
    db.refresh(db_preset)
    return db_preset

def get_user_presets_by_user(db: Session, user_id: uuid.UUID) -> List[db_models.UserPreset]:
    return db.query(db_models.UserPreset).filter(db_models.UserPreset.user_id == user_id).order_by(db_models.UserPreset.created_at.desc()).all()

def get_user_preset(db: Session, preset_id: uuid.UUID, user_id: uuid.UUID) -> Optional[db_models.UserPreset]:
    return db.query(db_models.UserPreset).filter(db_models.UserPreset.id == preset_id, db_models.UserPreset.user_id == user_id).first()

def update_user_preset(db: Session, db_preset: db_models.UserPreset, preset_data: preset_schemas.PresetUpdate) -> db_models.UserPreset:
    """Updates a preset record from a Pydantic schema."""
    update_data = preset_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_preset, key, value)
    db.commit()
    db.refresh(db_preset)
    return db_preset

def delete_user_preset(db: Session, db_preset: db_models.UserPreset) -> None:
    """Deletes a preset record from the database."""
    db.delete(db_preset)
    db.commit()