import logging
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import HTTPException, status
from typing import Optional # Added Optional for type hints

# Import configurations from config.py
try:
    from config import AWS_S3_BUCKET_NAME, AWS_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
except ImportError:
    # This is a fallback for cases where config.py might not be structured as expected
    # or for simpler testing environments. Production should rely on environment variables.
    AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "your-s3-bucket-name-fallback")
    AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID") # No default for credentials
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY") # No default for credentials

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.bucket_name = AWS_S3_BUCKET_NAME
        self.region = AWS_S3_REGION

        # Initialize S3 client
        # If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are None, boto3 will try to find credentials
        # through other methods (e.g., IAM roles, shared credentials file).
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            # Verify connection or credentials by making a simple call, e.g., head_bucket
            # This helps to catch credential or bucket access issues early.
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name} in region: {self.region}")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"S3 Configuration Error: Credentials not found or incomplete. {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="S3 credentials not configured.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.error(f"S3 Configuration Error: Bucket '{self.bucket_name}' does not exist or permission denied.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 bucket '{self.bucket_name}' not found or access denied.")
            elif e.response['Error']['Code'] == 'InvalidAccessKeyId' or e.response['Error']['Code'] == 'SignatureDoesNotMatch':
                logger.error(f"S3 Configuration Error: Invalid AWS credentials. {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid S3 credentials provided.")
            else:
                logger.error(f"S3 Configuration Error: Could not connect to S3. {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not connect to S3: {e}")
        except Exception as e: # Catch any other initialization errors
            logger.error(f"Unexpected error initializing S3 client: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error initializing S3 client: {e}")


    def upload_file(self, file_obj, object_key: str, content_type: Optional[str] = None, acl: str = "private") -> str:
        extra_args = {'ACL': acl}
        if content_type:
            extra_args['ContentType'] = content_type

        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Successfully uploaded {object_key} to S3 bucket {self.bucket_name} with ACL {acl}.")
            return object_key
        except ClientError as e:
            logger.error(f"S3 Upload Error: Failed to upload {object_key} to S3. Error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file to S3: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload of {object_key}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during S3 upload: {e}")

    def download_file(self, object_key: str, destination_path: str) -> None:
        try:
            self.s3_client.download_file(self.bucket_name, object_key, destination_path)
            logger.info(f"Successfully downloaded {object_key} from S3 to {destination_path}.")
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.error(f"S3 Download Error: File not found in S3: {object_key}. Error: {e}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found in S3: {object_key}")
            else:
                logger.error(f"S3 Download Error: Failed to download {object_key} from S3. Error: {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to download file from S3: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 download of {object_key}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during S3 download: {e}")

    def get_public_url(self, object_key: str) -> str:
        if self.region and self.region != "us-east-1":
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{object_key}"
        return f"https://{self.bucket_name}.s3.amazonaws.com/{object_key}"


    def generate_presigned_url(self, object_key: str, expiration_seconds: int = 3600) -> Optional[str]:
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration_seconds
            )
            logger.info(f"Generated presigned URL for {object_key} (expires in {expiration_seconds}s).")
            return response
        except ClientError as e:
            logger.error(f"S3 Presigned URL Error: Could not generate presigned URL for {object_key}. Error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not generate presigned URL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating presigned URL for {object_key}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error generating presigned URL: {e}")


    def delete_file(self, object_key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Successfully deleted {object_key} from S3 bucket {self.bucket_name} (or it did not exist).")
            return True
        except ClientError as e:
            logger.error(f"S3 Delete Error: Failed to delete {object_key} from S3. Error: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete file from S3: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during S3 deletion of {object_key}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error during S3 deletion: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Attempting to initialize StorageService...")
    try:
        storage = StorageService()
        logger.info("StorageService initialized.")
        from io import BytesIO
        dummy_file_content = b"This is a test file for S3."
        dummy_file_obj = BytesIO(dummy_file_content)
        test_object_key = "test_uploads/dummy_test_file.txt"
    except HTTPException as e:
        print(f"HTTPException during StorageService test: {e.detail}")
    except Exception as e:
        print(f"Generic error during StorageService test: {e}")
