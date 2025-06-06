from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session # Added Session
from typing import List # Added List

from models.models import User, UserSchema, EnhancementHistorySchema # Added EnhancementHistorySchema
from auth_utils import get_current_user
from db import crud # Added crud
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
