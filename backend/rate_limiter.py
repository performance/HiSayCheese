# rate_limiter.py
import logging
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM

logger = logging.getLogger(__name__)

# --- Step 1: Define a Key Function ---
# This function's only job is to return a unique identifier for the current request.
# It will be either the user's ID (from a JWT) or their IP address.
def get_request_identifier(request: Request) -> str:
    """
    Identifies the requester. If a valid JWT is present, it uses the user_id.
    Otherwise, it falls back to the client's IP address.
    """
    try:
        # Check for Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header:
            scheme, token = auth_header.split()
            if scheme.lower() == "bearer" and token:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                # The 'sub' claim is a standard for the subject (user identifier)
                user_id = payload.get("sub") or payload.get("user_id")
                if user_id:
                    # Store this key on the request state so our middleware can use it later
                    request.state.rate_limit_key = f"user:{user_id}"
                    return f"user:{user_id}"
    except (JWTError, ValueError, Exception):
        # Fallback for invalid token, no token, or other errors
        pass
    
    # Fallback to IP address for anonymous users
    ip_address = get_remote_address(request)
    request.state.rate_limit_key = f"ip:{ip_address}"
    return f"ip:{ip_address}"

# --- Step 2: Initialize the Limiter with the Key Function ---
limiter = Limiter(key_func=get_request_identifier, strategy="moving-window")

# --- Step 3: Define Rate Limits as Simple Constants ---
# These are the rate limit strings. No function is needed here anymore.
AUTH_USER_RATE_LIMIT = "100/minute"
ANON_USER_RATE_LIMIT = "20/minute"

# --- Step 4: Define the Dynamic Limit Function ---
# This function now correctly inspects the identifier returned by our key_func.
def get_dynamic_rate_limit(key: str) -> str:
    """
    Returns the appropriate rate limit string based on the identifier.
    'identifier' will be something like "user:some_uuid" or "ip:127.0.0.1".
    """
    if key.startswith("user:"):
        return AUTH_USER_RATE_LIMIT
    else: # It's an IP address
        return ANON_USER_RATE_LIMIT