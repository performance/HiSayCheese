import uuid # Added for path parameters
from fastapi import APIRouter, Depends, HTTPException, status, Response # Added HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List, Optional # Optional might be needed for response models if applicable

from models.models import User, UserSchema, EnhancementHistorySchema, PresetCreate, PresetSchema, PresetUpdate # Added Preset schemas
from auth_utils import get_current_user
from db import crud
from db.database import get_db # Added get_db


router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user.
    """
    return current_user


@router.get("/history", response_model=List[EnhancementHistorySchema])
async def read_user_enhancement_history(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get enhancement history for the current authenticated user.
    """
    history_records = crud.get_enhancement_history_by_user(
        db=db, user_id=current_user.id, skip=skip, limit=limit
    )
    return history_records


# User Preset Endpoints

@router.post("/presets", response_model=PresetSchema, status_code=status.HTTP_201_CREATED)
async def create_preset(
    preset_data: PresetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new preset for the current user.
    """
    return crud.create_user_preset(db=db, preset_data=preset_data, user_id=current_user.id)

@router.get("/presets", response_model=List[PresetSchema])
async def list_presets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all presets for the current user.
    """
    return crud.get_user_presets_by_user(db=db, user_id=current_user.id)

@router.get("/presets/{preset_id}", response_model=PresetSchema)
async def get_preset(
    preset_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific preset by its ID.
    """
    db_preset = crud.get_user_preset(db=db, preset_id=preset_id, user_id=current_user.id)
    if db_preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    return db_preset

@router.put("/presets/{preset_id}", response_model=PresetSchema)
async def update_preset(
    preset_id: uuid.UUID,
    preset_data: PresetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing preset.
    """
    updated_preset = crud.update_user_preset(db=db, preset_id=preset_id, preset_data=preset_data, user_id=current_user.id)
    if updated_preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found or not owned by user")
    return updated_preset

@router.delete("/presets/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preset(
    preset_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a preset.
    """
    deleted_preset = crud.delete_user_preset(db=db, preset_id=preset_id, user_id=current_user.id)
    if deleted_preset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found or not owned by user")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
