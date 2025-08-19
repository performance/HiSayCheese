from slowapi import Limiter
from limits.util import parse_many # Correct location for parse_many from the 'limits' library
from starlette.requests import Request
from typing import Optional, Any
from fastapi import Depends
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
                    identifier_type = "token_no_id" # Token was present but no user_id or sub
            # Key is combination of type and identifier
            key_to_return = f"{identifier_type}:{identifier}"
            if hasattr(request, 'state'): # Store for potential use in response headers via middleware
                request.state.rate_limit_key = key_to_return
            return key_to_return
    except JWTError:
        # logger.warning(f"JWTError decoding token for rate limit: {e}. Falling back to IP.")
        pass # Fall through to IP-based key
    except Exception: # Catch any other unexpected errors
        # logger.error(f"Unexpected error in get_request_identifier_for_rate_limit: {e}. Falling back to IP.", exc_info=True)
        pass # Fall through to IP-based key

    # Fallback to IP-based key
    key_to_return = f"ip:{identifier}"
    if hasattr(request, 'state'):
        request.state.rate_limit_key = key_to_return
    return key_to_return

limiter = Limiter(key_func=get_request_identifier_for_rate_limit)

# MODIFIED FUNCTION:
# This function is called by slowapi with the *key string* (e.g., "user:123" or "ip:1.2.3.4")
# that was returned by the 'key_func' (get_request_identifier_for_rate_limit).
def get_dynamic_rate_limit(key: str) -> str:  # Signature changed to accept the key string

    # logger.debug(f"Dynamic rate limit check. Key: {key}")

    if key.startswith("user:") or key.startswith("sub:") or key.startswith("token_no_id:"):
        # logger.debug(f"Applying authenticated user rate limit for key: {key}")
        return AUTH_USER_RATE_LIMIT
    elif key.startswith("ip:"):
        # logger.debug(f"Applying anonymous user rate limit for key: {key}")
        return ANON_USER_RATE_LIMIT
    else:
        # This case should ideally not be reached if key_func always prefixes correctly.
        logger.warning(f"Unexpected key format '{key}' for dynamic rate limit, applying anonymous limit.")
        return ANON_USER_RATE_LIMIT
