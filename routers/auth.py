from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from fastapi.security import OAuth2PasswordRequestForm
from db.database import get_db
from db import crud
from models.models import UserCreate, UserSchema # User model itself is used via crud.get_user_by_email
from auth_utils import verify_password, create_access_token

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Password Strength Check
    if len(user.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters long.",
        )

    # Check for existing user
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user (hashing is done within crud.create_user)
    created_user = crud.create_user(db=db, user=user)
    return created_user

@router.post("/login") # Or use "/token" for OAuth2 convention
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Ensure user.id is a string for JWT if it's a UUID object
    user_id_str = str(user.id)

    access_token_data = {"sub": user.email, "user_id": user_id_str}
    access_token = create_access_token(data=access_token_data)

    return {"access_token": access_token, "token_type": "bearer"}
