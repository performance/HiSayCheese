from slowapi import Limiter
from limits.util import parse_many # Correct location for parse_many from the 'limits' library
from starlette.requests import Request
from typing import Optional, Any
from fastapi import Depends # Added Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from config import SECRET_KEY, ALGORITHM # Assuming config.py is in the root or accessible
import logging

logger = logging.getLogger(__name__)
oauth2_scheme_optional = HTTPBearer(auto_error=False)

# Define rate limit strings - these could also be in config.py
AUTH_USER_RATE_LIMIT = "100/minute"
ANON_USER_RATE_LIMIT = "20/minute"

def get_request_identifier_for_rate_limit(request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme_optional)) -> str:
    identifier_type = "ip"
    identifier = request.client.host if request.client else "unknown_client" # Handle missing client
    try:
        if creds and creds.credentials:
            token = creds.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("user_id")
            if user_id:
                identifier = str(user_id)
                identifier_type = "user"
            else:
                sub = payload.get("sub")
                if sub:
                    identifier = str(sub)
                    identifier_type = "sub"
                else:
                    # If token is present but no user_id or sub, it's an issue, but still log with token prefix
                    # to distinguish from purely anonymous. Or treat as anonymous.
                    # For now, let's make it distinct if a token was attempted.
                    identifier_type = "token_no_id"
                    # logger.warning("Token present but no 'user_id' or 'sub' claim found. Falling back to IP for rate limiting key, but type is 'token_no_id'.")
                    # The key itself will still be IP-based if no better identifier found from token.
                    # The goal is to return a string key for the limiter.
            # Ensure the key is always prefixed for clarity in dynamic limit logic
            return f"{identifier_type}:{identifier}"
    except JWTError: # Specific error for JWT issues
        # logger.warning(f"JWTError during token decoding for rate limit key: {e}. Falling back to IP.")
        # Fallback to IP, but mark that a token was attempted and failed.
        # This helps differentiate from purely anonymous requests if needed for debugging.
        # However, for rate limiting purposes, might be better to treat as anonymous.
        # Let's stick to IP for now if token processing fails.
        pass # Fall through to IP-based key
    except Exception: # Catch any other unexpected errors
        # logger.error(f"Unexpected error in get_request_identifier_for_rate_limit: {e}. Falling back to IP.", exc_info=True)
        pass # Fall through to IP-based key

    # Fallback to IP if no token or token processing failed
    # Ensure request.state.rate_limit_key is set for the header middleware
    # This part is tricky as request.state might not be available when key_func is called by limiter globally
    # if it's not within a request cycle where middleware has added it.
    # However, for slowapi, key_func is called per request.
    key = f"ip:{identifier}" # identifier here is request.client.host
    if hasattr(request, 'state'):
        request.state.rate_limit_key = key
    # logger.debug(f"Rate limiting by {key} for request to {request.url.path}") # Can be verbose
    return key

limiter = Limiter(key_func=get_request_identifier_for_rate_limit)

def get_dynamic_rate_limit(key_or_request: Any) -> str: # Argument can be key (str) or Request
    # The key function of slowapi passes the request object to this callable if it's set as a limit value.
    # If a string is returned by key_func, that string is passed here.
    # The Limiter instance's key_func returns a string like "ip:127.0.0.1" or "user:some_id"
    # So, the argument here will be that string key.

    key_str = ""
    if isinstance(key_or_request, str):
        key_str = key_or_request
    elif isinstance(key_or_request, Request): # Should not happen with current Limiter setup
        # If key_func was returning the request, we'd call it here.
        # But our key_func returns a string.
        # For safety, handle if request object is passed, though not expected.
        logger.warning("get_dynamic_rate_limit received Request object, expected string key.")
        key_str = get_request_identifier_for_rate_limit(key_or_request) # Recalculate key
    else:
        logger.error(f"get_dynamic_rate_limit received unexpected type: {type(key_or_request)}. Defaulting to anonymous limit.")
        return ANON_USER_RATE_LIMIT

    if key_str.startswith("user:") or key_str.startswith("sub:") or key_str.startswith("token_no_id:"): # Treat "token_no_id" as authenticated for rate limit purposes
        # logger.debug(f"Applying authenticated user rate limit for key: {key_str}")
        return AUTH_USER_RATE_LIMIT
    elif key_str.startswith("ip:"):
        # logger.debug(f"Applying anonymous user rate limit for key: {key_str}")
        return ANON_USER_RATE_LIMIT
    else: # Default for unexpected key formats
        logger.warning(f"Unexpected key format '{key_str}' for dynamic rate limit, applying anonymous limit.")
        return ANON_USER_RATE_LIMIT
