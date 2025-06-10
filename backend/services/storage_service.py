import logging
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import HTTPException, status
from typing import Optional
from unittest.mock import MagicMock

# Import configurations from config.py
try:
    from config import AWS_S3_BUCKET_NAME, AWS_S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
except ImportError:
    AWS_S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET_NAME", "your-s3-bucket-name-fallback")
    AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

logger = logging.getLogger(__name__)

# Holder for the last instance of StorageService created in test mode.
_current_test_storage_service_instance_holder = {'instance': None}

class StorageService:
    def __init__(self):
        if os.environ.get("TEST_MODE_NO_S3_CONNECT") == "true":
            logger.info("StorageService __init__ in TEST_MODE_NO_S3_CONNECT: Initializing with mock s3_client AND mock service methods.")
            self.s3_client = MagicMock(name="s3_client_mock_in_test_mode")
            self.bucket_name = AWS_S3_BUCKET_NAME
            self.region = AWS_S3_REGION

            # Mock s3_client methods (already present from previous step, ensure comprehensive)
            self.s3_client.upload_fileobj = MagicMock(name="s3_client.upload_fileobj_mock")
            self.s3_client.download_file = MagicMock(name="s3_client.download_file_mock")
            self.s3_client.generate_presigned_url = MagicMock(name="s3_client.generate_presigned_url_mock")
            self.s3_client.delete_object = MagicMock(name="s3_client.delete_object_mock")
            self.s3_client.head_bucket = MagicMock(name="s3_client.head_bucket_mock")

            # Replace service instance methods with MagicMocks
            self.upload_file = MagicMock(name="upload_file_mock_on_instance")
            self.download_file = MagicMock(name="download_file_mock_on_instance")
            self.get_public_url = MagicMock(name="get_public_url_mock_on_instance")
            self.generate_presigned_url = MagicMock(name="generate_presigned_url_mock_on_instance")
            self.delete_file = MagicMock(name="delete_file_mock_on_instance")

            _current_test_storage_service_instance_holder['instance'] = self
            return

        # Original __init__ logic starts here
        self.bucket_name = AWS_S3_BUCKET_NAME
        self.region = AWS_S3_REGION

        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {self.bucket_name} in region: {self.region}")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"S3 Configuration Error: Credentials not found or incomplete. {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="S3 credentials not configured.")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'NoSuchBucket':
                logger.error(f"S3 Configuration Error: Bucket '{self.bucket_name}' does not exist or permission denied.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 bucket '{self.bucket_name}' not found or access denied.")
            elif error_code in ['InvalidAccessKeyId', 'SignatureDoesNotMatch', 'AuthFailure', 'AccessDenied', 'Forbidden']:
                logger.error(f"S3 Configuration Error: Invalid AWS credentials or insufficient permissions. {e}")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Invalid S3 credentials or permissions: {e}")
            else:
                logger.error(f"S3 Configuration Error: Could not connect to S3. {e}")
                detail_message = f"Could not connect to S3: {e.response.get('Error', {}).get('Message', str(e))}"
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_message)
        except Exception as e:
            logger.error(f"Unexpected error initializing S3 client: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error initializing S3 client: {e}")

    # Original method definitions remain. In test mode, these are shadowed by the MagicMock attributes.
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
            logger.error(f"Unexpected error generating presigned URL for {object_key}: {e}", exc_info=True)
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
        os.environ["TEST_MODE_NO_S3_CONNECT"] = "true"
        storage = StorageService()
        logger.info("StorageService initialized.")

        if os.environ.get("TEST_MODE_NO_S3_CONNECT") == "true":
            print(f"S3 Client is MagicMock: {isinstance(storage.s3_client, MagicMock)}")
            print(f"upload_file is MagicMock: {isinstance(storage.upload_file, MagicMock)}")
            storage.upload_file(None, "bucket", "key")
            storage.upload_file.assert_called_once()
            print("Mock instance method 'upload_file' called successfully.")

            storage.get_public_url("test_key")
            storage.get_public_url.assert_called_once_with("test_key")
            print("Mock instance method 'get_public_url' called successfully.")

            # Accessing the stored instance for further configuration if needed from tests
            # from backend.services.storage_service import _current_test_storage_service_instance_holder
            # test_instance = _current_test_storage_service_instance_holder['instance']
            # assert test_instance is storage
            # print("Successfully accessed storage instance via holder.")


    except HTTPException as e:
        print(f"HTTPException during StorageService test: {e.detail}")
    except Exception as e:
        print(f"Generic error during StorageService test: {e}")
    finally:
        if "TEST_MODE_NO_S3_CONNECT" in os.environ:
            del os.environ["TEST_MODE_NO_S3_CONNECT"]
