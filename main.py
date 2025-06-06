# main.py
import uuid
import magic
import os # Added os import
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status # Added status
from sqlalchemy.orm import Session
# JSONResponse is not strictly needed if returning Pydantic model with status_code
# from fastapi.responses import JSONResponse

from db.database import create_db_and_tables, get_db
from db import crud
from models import models # This imports the models module
# We need ImageCreate and ImageSchema for type hinting and response_model
# from models.models import ImageCreate, ImageSchema # More specific imports
import logging # Added logging

app = FastAPI()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads/images/"
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"]
MIME_TYPE_TO_EXTENSION = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/health", response_model=models.NumberSchema) # Corrected to NumberSchema
async def health(db: Session = Depends(get_db)):
    db_number = crud.get_number(db)
    if db_number is None:
        return JSONResponse(status_code=404, content={"message": "No number set yet"})
    return db_number

@app.post("/put_number", response_model=models.NumberSchema) # Corrected to NumberSchema
async def put_number(number: models.NumberCreate, db: Session = Depends(get_db)):
    return crud.create_or_update_number(db=db, value=number.value)

@app.post("/increment_number", response_model=models.NumberSchema) # Corrected to NumberSchema
async def increment_number_endpoint(db: Session = Depends(get_db)):
    db_number = crud.increment_number(db)
    if db_number is None:
        raise HTTPException(status_code=404, detail="No number set to increment.")
    return db_number

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Placeholder malware scan function
async def scan_for_malware(contents: bytes) -> bool:
    """
    Placeholder for malware scanning logic.
    In a real application, this would involve integrating with a malware scanning engine.
    """
    logger.info("Malware scan stub: assuming file is safe.")
    # Simulate some async work if needed, e.g., await asyncio.sleep(0.01)
    return True

# Updated image upload endpoint
@app.post("/api/images/upload", response_model=models.ImageSchema, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    file_size = len(contents)

    # 1. File Size Validation
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413, # Payload Too Large
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # 2. File Type Validation
    # Initial check based on browser-provided content type (file.content_type)
    # More robust check using python-magic
    detected_mime_type = magic.from_buffer(contents, mime=True)

    if detected_mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{detected_mime_type}'. Allowed types are JPG, PNG, WEBP."
        )

    actual_content_type = detected_mime_type
    file_extension = MIME_TYPE_TO_EXTENSION.get(actual_content_type)

    if not file_extension:
        # This should ideally not happen if ALLOWED_MIME_TYPES and MIME_TYPE_TO_EXTENSION are synced
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File type processed correctly, but no extension mapping found."
        )

    # 3. Malware Scan (Placeholder)
    # This is called after initial validation but before saving to disk or DB
    if not await scan_for_malware(contents):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malware detected in file."
        )

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate unique filename
    unique_filename_base = uuid.uuid4().hex
    unique_filename = f"{unique_filename_base}{file_extension}"
    server_filepath = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the file
    try:
        with open(server_filepath, "wb") as f:
            f.write(contents)
    except IOError as e:
        # Log the exception e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {e}",
        )

    # File has been read for validation and then for saving.
    # If contents were not stored in a variable, file.seek(0) and file.read() would be needed here.
    # But since `contents` holds the data, we can use it directly.

    # Create database record
    image_data_to_create = models.ImageCreate(
        filename=file.filename, # Original filename
        filepath=server_filepath,
        filesize=file_size,
        mimetype=actual_content_type
    )

    db_image = crud.create_image(db=db, image=image_data_to_create)

    # The response_model=models.ImageSchema will automatically handle the conversion
    return db_image
