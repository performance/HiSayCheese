# main.py
# The file is now solely responsible for:
# Creating the FastAPI app instance.
# Configuring and adding middleware (CORS, Request ID, Logging, Security, Rate Limiting).
# Setting up global services like logging and Sentry.
# Handling application lifecycle events (on_startup).
# Including the various Routers from the routers/ directory, which now contain the actual endpoint logic.


# main.py
import uuid
import time
import logging
from pythonjsonlogger import jsonlogger
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest # Renamed to avoid shadowing
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from jose import jwt, JWTError
from fastapi.security import HTTPBearer

# --- Rate Limiting Imports ---
# These are needed for the middleware and exception handler defined in this file.
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from limits.util import parse_many
from rate_limiter import limiter, get_dynamic_rate_limit

# --- Local Project Imports ---
from config import SECRET_KEY, ALGORITHM, SENTRY_DSN
from db.database import create_db_and_tables
from routers import auth as auth_router
from routers import users as users_router
from routers import health as health_router
from routers import images as images_router
from routers import analysis as analysis_router
from routers import enhancement as enhancement_router

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# --- Type Hinting ---
from typing import Callable, Awaitable, Any

RequestResponseCall = Callable[[StarletteRequest], Awaitable[StarletteResponse]]


# --- Logging Configuration ---
# Configure this early so all subsequent modules can use it.
log_handler = logging.StreamHandler()
# This formatter will be dynamically updated by the RequestIdMiddleware to add request-specific fields.
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s '
    '%(request_id)s %(user_id)s %(path)s %(method)s %(status_code)s %(response_time_ms)s'
)
log_handler.setFormatter(formatter)

# Configure the root logger to capture logs from all libraries (e.g., sqlalchemy, uvicorn)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any default handlers to avoid duplicate logs
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(log_handler)

# Get a logger instance for our application's own logs
logger = logging.getLogger(__name__)


# --- Sentry Initialization ---
if SENTRY_DSN and SENTRY_DSN != "your-sentry-dsn-goes-here":
    sentry_logging = LoggingIntegration(
        level=logging.DEBUG,        # Breadcrumbs level
        event_level=logging.INFO    # Event level (INFO and above)
    )
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[sentry_logging],
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0
    )
    logger.info("Sentry initialized.")
else:
    logger.warning("Sentry DSN not found or is a placeholder. Sentry will not be initialized.")


# --- Constants ---
# Constants for middleware configuration. Endpoint-specific constants were moved to their respective routers.
MAX_REQUEST_BODY_SIZE = 1 * 1024 * 1024  # 1MB for general JSON bodies


# --- Middleware Definitions ---

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCall) -> StarletteResponse:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id

        user_id_for_log = "anonymous"
        try:
            token_creds = await HTTPBearer(auto_error=False)(request)
            if token_creds and token_creds.credentials:
                payload = jwt.decode(token_creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
                user_id_for_log = payload.get("user_id") or payload.get("sub")
        except (JWTError, Exception):
            pass  # Token is invalid, not present, or expired. Fine for logging.
        
        request.state.user_id = user_id_for_log

        original_factory = logging.getLogRecordFactory()

        def new_log_record_factory(*args, **kwargs):
            record = original_factory(*args, **kwargs)
            record.request_id = getattr(request.state, 'request_id', 'N/A')
            record.user_id = getattr(request.state, 'user_id', None)
            return record

        logging.setLogRecordFactory(new_log_record_factory)
        
        try:
            response = await call_next(request)
        finally:
            logging.setLogRecordFactory(original_factory) # Ensure reset

        response.headers['X-Request-ID'] = request_id
        return response

class ResponseTimeLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCall) -> StarletteResponse:
        start_time = time.time()
        response = await call_next(request)
        process_time_ms = (time.time() - start_time) * 1000

        # The log record factory in RequestIdMiddleware will add request_id and user_id.
        # We just add the fields specific to this middleware.
        log_details = {
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "response_time_ms": round(process_time_ms, 2)
        }
        logger.info("Request processed", extra=log_details)
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCall) -> StarletteResponse:
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "default-src 'self'; object-src 'none'; frame-ancestors 'none';"
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

# The corrected version that also skips the login form endpoint
class RequestBodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_size: int):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCall) -> StarletteResponse:
        # Define a set of paths to skip this middleware for.
        # These paths handle streaming or special body types (forms, files) themselves.
        # This is the most robust way to handle exclusions.
        excluded_paths = [
            "/api/images/upload",
            "/api/auth/login", 
            # Add any other form or file upload paths here in the future
        ]

        if request.url.path in excluded_paths:
             logger.debug(f"Skipping RequestBodySizeLimitMiddleware for path: {request.url.path}")
             return await call_next(request)

        # The rest of the middleware logic for handling JSON bodies remains the same...
        content_length_header = request.headers.get("content-length")
        if content_length_header and int(content_length_header) > self.max_size:
            msg = f"Request body size ({content_length_header} bytes) exceeds limit of {self.max_size} bytes."
            logger.warning(msg)
            return StarletteResponse(content=msg, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

        # To handle chunked encoding without content-length
        body = b''
        async for chunk in request.stream():
            body += chunk
            if len(body) > self.max_size:
                msg = f"Request body stream exceeded limit of {self.max_size} bytes."
                logger.warning(msg)
                return StarletteResponse(content=msg, status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        
        async def new_receive():
            return {"type": "http.request", "body": body}
        
        request.scope['receive'] = new_receive
        return await call_next(request)

class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseCall) -> StarletteResponse:
        response = await call_next(request)
        
        # The limiter instance is on app.state
        limiter_instance: Limiter = request.app.state.limiter
        
        # The key is now correctly set on the request state by our key function
        key = getattr(request.state, 'rate_limit_key', None)

        if not key:
            return response

        try:
            # slowapi.limiter has a method to get the limits for a key
            # This is the new, correct way to do this.
            # We don't need to access .storage directly.
            # The check() method returns a tuple of (is_limited, remaining, reset_time)
            # This is not what we want here; we want to get the *current* window stats.
            
            # The best way is to interact with the limiter's internal storage if we must.
            # A common pattern is that the `limiter` object itself has methods to help.
            # Let's check slowapi documentation... `limiter.check()` and `limiter.hit()` are the main ones.
            # A simpler way to get stats might not be exposed.
            
            # Let's try a different approach. The library's main purpose is to raise an exception.
            # The custom exception handler is a more reliable place to get this data.
            # For successful requests, the headers are less critical.
            # Let's simplify this middleware to avoid relying on unstable internals.
            # We will focus on adding headers only to the 429 response, which is more important.
            
            # --- SIMPLIFIED AND ROBUST APPROACH ---
            # We can't easily get the 'remaining' count without hitting the limit.
            # So, we will only add the 'X-RateLimit-Limit' header for successful requests.
            rate_string = get_dynamic_rate_limit(key)
            limit_list = parse_many(rate_string)
            if limit_list:
                limit_obj = limit_list[0]
                response.headers["X-RateLimit-Limit"] = str(limit_obj.amount)

        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}", exc_info=True)
            
        return response


# The custom exception handler is the most reliable place to get limit info.
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler to add rate limit headers to 429 responses."""
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )
    
    # The 'RateLimitExceeded' exception object now contains the necessary info.
    window_stats = exc.meta.get("window_stats")
    if window_stats:
        reset_time = window_stats[0]
        limit_amount = window_stats[1]
        
        response.headers["X-RateLimit-Limit"] = str(limit_amount)
        response.headers["X-RateLimit-Remaining"] = "0"
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
    else:
        logger.warning("Could not add rate limit headers to 429 response: window_stats not found in exception.")
        
    return response

# --- FastAPI Application Setup ---

# --- Application Events ---

# @app.on_event("startup")
# def on_startup():
#     logger.info("Application starting up...")
#     create_db_and_tables()
#     logger.info("Database tables checked/created.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Application starting up...")
    create_db_and_tables()
    logger.info("Database tables checked/created.")
    print("Database tables checked/created.")
    yield
    # Code to run on shutdown
    print("Application shutting down.")




app = FastAPI(
    lifespan=lifespan,
    title="Image Enhancement API",
    description="An API for uploading, analyzing, and enhancing portrait images.",
    version="1.0.0"
)

# --- Mount static files directory for local storage mode ---
# This line is crucial for making local file storage accessible via URL.
# It creates a special route `/static` that serves files from the `local_storage` directory.
# The check_dir=False is important because the directory might not exist at startup,
# but the StorageService will create it.
app.mount("/static", StaticFiles(directory=Path("local_storage")), name="static")


# Add Rate Limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)

# Add Middlewares in order of execution (first added is outermost)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ResponseTimeLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
# app.add_middleware(RequestBodySizeLimitMiddleware, max_size=MAX_REQUEST_BODY_SIZE)
app.add_middleware(RateLimitHeaderMiddleware)



# --- Include Routers ---
# All endpoint logic is now in these router files.
app.include_router(auth_router.router, tags=["Authentication"])
app.include_router(users_router.router, tags=["Users"])
app.include_router(health_router.router, tags=["Health"])
app.include_router(images_router.router, tags=["Image Upload"])
app.include_router(analysis_router.router, tags=["Image Analysis"])
app.include_router(enhancement_router.router, tags=["Image Enhancement"])


# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Image Enhancement API is running."}