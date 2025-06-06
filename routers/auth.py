from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
import logging # Added for logging
from datetime import datetime # Added for token expiry check

from fastapi.security import OAuth2PasswordRequestForm
from db.database import get_db
from db import crud
from models.models import UserCreate, UserSchema # User model itself is used via crud.get_user_by_email
from auth_utils import verify_password, create_access_token
from ..main import limiter, get_dynamic_rate_limit # Import limiter and the dynamic rate limit function

# Import EmailService and config
from services.email_service import EmailService
from config import FRONTEND_URL

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Instantiate EmailService globally
try:
    email_service = EmailService()
    logger.info("EmailService initialized successfully in auth_router.")
except HTTPException as e:
    logger.error(f"Auth Router: Failed to initialize EmailService (HTTPException): {e.detail}")
    email_service = None
except Exception as e:
    logger.error(f"Auth Router: Failed to initialize EmailService (General Exception): {e}", exc_info=True)
    email_service = None # Corrected variable name

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    responses={404: {"description": "Not found"}},
)

@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
@limiter.limit(get_dynamic_rate_limit) # Use the dynamic rate limit function
def register_user(
    request: Request,
    user: UserCreate,
    background_tasks: BackgroundTasks, # Added BackgroundTasks
    db: Session = Depends(get_db)
):
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

    # Send verification email in the background
    if email_service and created_user.verification_token:
        verification_url = f"{FRONTEND_URL}/auth/verify-email?token={created_user.verification_token}"
        try:
            email_content = email_service.get_verification_email_template(
                username=created_user.email, # Using email as username for template
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
        except Exception as e_template:
            # Handle error in getting email template (though unlikely with current static template)
            logger.error(f"Failed to get verification email template for user {created_user.email}: {e_template}", exc_info=True)
    elif not email_service:
        logger.error("EmailService not available. Verification email for {created_user.email} will not be sent.")
    elif not created_user.verification_token: # Should not happen if crud.create_user works as expected
        logger.error(f"User {created_user.email} created without a verification token. Cannot send verification email.")

    return created_user

@router.post("/login") # Or use "/token" for OAuth2 convention
@limiter.limit(get_dynamic_rate_limit) # Use the dynamic rate limit function
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
@limiter.limit(get_dynamic_rate_limit) # Apply rate limiting
async def verify_user_email(request: Request, token: str, db: Session = Depends(get_db)):
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
