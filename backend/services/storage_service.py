import logging
import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import HTTPException, status, Request

# Import the new config variable
from config import STORAGE_TYPE, AWS_S3_BUCKET_NAME, AWS_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

logger = logging.getLogger(__name__)

# Base directory for local file storage
LOCAL_STORAGE_PATH = Path("local_storage")
# Base URL for accessing local files (assuming FastAPI is running on localhost:8000)
# This will be used to construct URLs to serve the files.
# In a real setup, you might serve this static directory via Nginx.
LOCAL_URL_BASE = "/static" 

class StorageService:
    """
    A storage service that can operate in two modes:
    - 'local': Saves files to the local filesystem. Ideal for development.
    - 's3': Saves files to an AWS S3 bucket. For production.

    The mode is determined by the `STORAGE_TYPE` environment variable.
    """
    def __init__(self, request: Request = None):
        self.mode = STORAGE_TYPE
        self.request = request # Store the request object to build full URLs
        
        logger.info(f"Initializing StorageService in '{self.mode}' mode.")

        if self.mode == "local":
            self._init_local()
        elif self.mode == "s3":
            self._init_s3()
        else:
            raise ValueError(f"Invalid STORAGE_TYPE: '{self.mode}'. Must be 'local' or 's3'.")

    def _init_local(self):
        """Initializes the local storage directory."""
        # Create the main storage directory and subdirectories if they don't exist
        self.bucket_name = LOCAL_STORAGE_PATH
        self.bucket_name.mkdir(exist_ok=True)
        (self.bucket_name / "original_images").mkdir(exist_ok=True)
        (self.bucket_name / "processed_images").mkdir(exist_ok=True)
        logger.info(f"Local storage initialized at: ./{self.bucket_name}")

    def _init_s3(self):
        """Initializes the S3 client and verifies the connection."""
        self.bucket_name = AWS_S3_BUCKET_NAME
        self.region = AWS_S3_REGION

        if not all([self.bucket_name, self.region, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
            logger.error("S3 mode selected, but one or more required AWS environment variables are missing.")
            raise HTTPException(status_code=500, detail="S3 storage is not configured correctly.")
        
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise HTTPException(status_code=500, detail=f"Could not connect to S3. Check credentials and permissions. Error: {e}")

    def upload_file(self, file_obj: BinaryIO, object_key: str, **kwargs) -> str:
        if self.mode == 'local':
            return self._upload_file_local(file_obj, object_key)
        else: # s3
            return self._upload_file_s3(file_obj, object_key, **kwargs)

    def download_file(self, object_key: str, destination_path: str) -> None:
        if self.mode == 'local':
            self._download_file_local(object_key, destination_path)
        else: # s3
            self._download_file_s3(object_key, destination_path)
            
    def generate_presigned_url(self, object_key: str, expiration_seconds: int = 3600) -> Optional[str]:
        if self.mode == 'local':
            return self._generate_url_local(object_key)
        else: # s3
            return self._generate_presigned_url_s3(object_key, expiration_seconds)

    # --- Local Mode Implementations ---

    def _upload_file_local(self, file_obj: BinaryIO, object_key: str):
        file_path = self.bucket_name / object_key
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)
        logger.info(f"Saved file locally to: {file_path}")
        return object_key

    def _download_file_local(self, object_key: str, destination_path: str):
        source_path = self.bucket_name / object_key
        if not source_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found locally: {object_key}")
        shutil.copy(source_path, destination_path)
        logger.info(f"Copied local file from {source_path} to {destination_path}")

    def _generate_url_local(self, object_key: str) -> Optional[str]:
        if not self.request:
            logger.warning("Cannot generate full URL for local file without request context.")
            # Return a relative path as a fallback
            return f"{LOCAL_URL_BASE}/{object_key}"
        # Construct a full URL like http://127.0.0.1:8000/static/original_images/uuid.jpg
        return str(self.request.url.replace(path=f"{LOCAL_URL_BASE}/{object_key}"))

    # --- S3 Mode Implementations ---

    def _upload_file_s3(self, file_obj: BinaryIO, object_key: str, **kwargs):
        # ... (paste your existing S3 upload_file logic here, just renamed)
        try:
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, object_key, ExtraArgs=kwargs)
            return object_key
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    def _download_file_s3(self, object_key: str, destination_path: str):
        # ... (paste your existing S3 download_file logic here, just renamed)
        try:
            self.s3_client.download_file(self.bucket_name, object_key, destination_path)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                raise HTTPException(status_code=404, detail=f"File not found in S3: {object_key}")
            raise HTTPException(status_code=500, detail=f"S3 download failed: {e}")

    def _generate_presigned_url_s3(self, object_key: str, expiration_seconds: int):
        # ... (paste your existing S3 generate_presigned_url logic here, just renamed)
        try:
            return self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration_seconds
            )
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"Could not generate presigned S3 URL: {e}")