from fastapi import APIRouter, Depends
from models.models import User, UserSchema # User is the SQLAlchemy model, UserSchema is Pydantic
from auth_utils import get_current_user

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
