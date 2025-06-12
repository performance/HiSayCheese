from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.params import Body
from sqlalchemy.orm import Session
import logging # Added for logging
from datetime import datetime # Added for token expiry check

from fastapi.security import OAuth2PasswordRequestForm
from db.database import get_db
from db import crud
from schemas.user_schemas import UserCreate, UserSchema
from auth_utils import verify_password, create_access_token
from rate_limiter import limiter, get_dynamic_rate_limit # Import from new rate_limiter module

# Import EmailService and config
from services.email_service import EmailService
from config import FRONTEND_URL
from dependencies import get_email_service

# Initialize logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
# @limiter.limit(get_dynamic_rate_limit) # Use the dynamic rate limit function
def register_user(
    request: Request,
    background_tasks: BackgroundTasks, # Added BackgroundTasks
    user: UserCreate, # =Body(...),
    db: Session = Depends(get_db),
    email_service: EmailService = Depends(get_email_service)
):
    """
    Registers a new user, checks for duplicates, and sends a verification email.
    """
    logger.info(f"Registration attempt for email: {user.email}")
    # Password strength (including length) is now handled by Pydantic model UserCreate

    # Check for existing user
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user (hashing is done within crud.create_user)
    created_user = crud.create_user(db=db, user=user)
    logger.info(f"User created successfully with ID: {created_user.id}")

    # Send verification email in the background
    # The 'email_service' variable is now guaranteed to be a valid instance.
    verification_url = f"{FRONTEND_URL}/auth/verify-email?token={created_user.verification_token}"
    email_content = email_service.get_verification_email_template(
        username=created_user.email,
        verification_url=verification_url
    )
    background_tasks.add_task(
        email_service.send_email,
        to_address=created_user.email,
        subject=email_content["subject"],
        html_body=email_content["html_body"],
        text_body=email_content["text_body"]
    )
    logger.info(f"Verification email task added for user: {created_user.email}")
    
    return created_user

@router.post("/login") # Or use "/token" for OAuth2 convention
# @limiter.limit(get_dynamic_rate_limit) # Use the dynamic rate limit function
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
): # Removed request: Request
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
    # Note: Login does not check for is_verified status yet.
    # This might be a requirement for a future subtask.

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/verify-email", response_model=UserSchema)
# @limiter.limit(get_dynamic_rate_limit) # Apply rate limiting
async def verify_user_email(
    request: Request,
    token: str, 
    db: Session = Depends(get_db)
): # Removed request: Request
    logger.info(f"Attempting email verification with token: {token}")

    user = crud.get_user_by_verification_token(db, token=token)

    if not user:
        logger.warning(f"Email verification failed: Token not found or invalid token {token}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    if user.is_verified:
        logger.info(f"Email for user {user.email} (token: {token}) already verified.")
        # Optionally, redirect or return a specific message if already verified
        # For now, consistent error as if token is invalid to prevent info leakage
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified or token invalid.", # Slightly different message
        )

    if user.verification_token_expires_at is None or datetime.utcnow() > user.verification_token_expires_at:
        logger.warning(f"Verification token for user {user.email} (token: {token}) has expired.")
        # Optionally, you might want to allow resending a new token here, or just fail.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired.",
        )

    try:
        updated_user = crud.mark_user_as_verified(db, user=user)
        logger.info(f"User {updated_user.email} successfully verified.")
        return updated_user
    except Exception as e: # Catch potential DB errors during update
        logger.error(f"Error marking user {user.email} as verified: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while verifying your email. Please try again later.",
        )
