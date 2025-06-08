import unittest
from unittest.mock import patch, MagicMock, call
import os

# Set environment variables for moto BEFORE importing StorageService or config
# This ensures that StorageService picks up these dummy values during its import/init
os.environ["AWS_S3_BUCKET_NAME"] = "test-bucket"
os.environ["AWS_S3_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "testing" # For moto
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing" # For moto
os.environ["AWS_SESSION_TOKEN"] = "testing" # For moto, if assuming IAM role

from fastapi import HTTPException, status
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from io import BytesIO

# Now import the service, it will use the mocked env vars if config relies on them at import time
# However, StorageService reads from config.py, which reads os.getenv at import time.
# So the os.environ patching above should be effective.
from services.storage_service import StorageService

# If config.py has already been imported elsewhere and cached, the above os.environ might not take effect
# for config.py itself. To be extremely robust for testing, one might need to reload config or
# directly patch the config module's variables if StorageService imports them directly.
# For this example, we assume os.environ is set early enough.


class TestStorageService(unittest.TestCase):

    def setUp(self):
        # This ensures that each test gets a fresh StorageService instance
        # and that os.environ changes are localized if we were to change them per test.
        # The critical part is that boto3.client is patched.
        pass

    @patch('boto3.client')
    def test_init_successful(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        service = StorageService()

        mock_boto_client.assert_called_once_with(
            "s3",
            aws_access_key_id="testing", # From our mocked os.environ
            aws_secret_access_key="testing",
            region_name="us-east-1"
        )
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")
        self.assertEqual(service.bucket_name, "test-bucket")
        self.assertEqual(service.region, "us-east-1")

    @patch('boto3.client')
    def test_init_no_credentials_error(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = NoCredentialsError()

        with self.assertRaises(HTTPException) as cm:
            StorageService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("S3 credentials not configured", cm.exception.detail)

    @patch('boto3.client')
    def test_init_partial_credentials_error(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.side_effect = PartialCredentialsError(provider="aws", cred_var="secret_key")

        with self.assertRaises(HTTPException) as cm:
            StorageService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("S3 credentials not configured", cm.exception.detail)

    @patch('boto3.client')
    def test_init_client_error_no_such_bucket(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        error_response = {'Error': {'Code': 'NoSuchBucket'}}
        mock_s3.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

        with self.assertRaises(HTTPException) as cm:
            StorageService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("S3 bucket 'test-bucket' not found or access denied", cm.exception.detail)

    @patch('boto3.client')
    def test_init_client_error_invalid_access_key(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        error_response = {'Error': {'Code': 'InvalidAccessKeyId'}}
        mock_s3.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

        with self.assertRaises(HTTPException) as cm:
            StorageService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Invalid S3 credentials provided", cm.exception.detail)

    @patch('boto3.client')
    def test_init_client_error_other(self, mock_boto_client):
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        error_response = {'Error': {'Code': 'SomeOtherError'}}
        mock_s3.head_bucket.side_effect = ClientError(error_response, "HeadBucket")

        with self.assertRaises(HTTPException) as cm:
            StorageService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Could not connect to S3", cm.exception.detail)

    @patch('services.storage_service.boto3.client') # Patch where boto3.client is used
    def test_upload_file_successful(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_boto_client_for_service.return_value = mock_s3_instance

        # Need to bypass the __init__ head_bucket call for this specific test,
        # or ensure it's also properly mocked if StorageService is instantiated per method.
        # For simplicity, we assume __init__ worked or patch head_bucket if needed.
        with patch.object(StorageService, '_StorageService__init__', lambda x: None): # Temporarily disable __init__
            service = StorageService() # Now __init__ does nothing
            service.s3_client = mock_s3_instance # Assign the mocked client
            service.bucket_name = "test-bucket" # Manually set bucket_name
            service.region = "us-east-1" # Manually set region

        file_obj = BytesIO(b"test data")
        object_key = "test/key.txt"

        result = service.upload_file(file_obj, object_key, content_type="text/plain", acl="public-read")

        self.assertEqual(result, object_key)
        mock_s3_instance.upload_fileobj.assert_called_once_with(
            file_obj,
            "test-bucket",
            object_key,
            ExtraArgs={'ACL': 'public-read', 'ContentType': 'text/plain'}
        )

    @patch('services.storage_service.boto3.client')
    def test_upload_file_client_error(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_s3_instance.upload_fileobj.side_effect = ClientError({'Error': {'Code': 'AccessDenied'}}, "UploadFileobj")
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        file_obj = BytesIO(b"test data")
        with self.assertRaises(HTTPException) as cm:
            service.upload_file(file_obj, "test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to upload file to S3", cm.exception.detail)

    @patch('services.storage_service.boto3.client')
    def test_download_file_successful(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        object_key = "test/key.txt"
        destination_path = "/tmp/test_download.txt"

        service.download_file(object_key, destination_path)

        mock_s3_instance.download_file.assert_called_once_with(
            "test-bucket",
            object_key,
            destination_path
        )

    @patch('services.storage_service.boto3.client')
    def test_download_file_client_error_404(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_s3_instance.download_file.side_effect = ClientError(error_response, "DownloadFile")
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        with self.assertRaises(HTTPException) as cm:
            service.download_file("test/nonexistent.txt", "/tmp/download.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("File not found in S3", cm.exception.detail)

    @patch('services.storage_service.boto3.client')
    def test_download_file_client_error_other(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        mock_s3_instance.download_file.side_effect = ClientError(error_response, "DownloadFile")
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        with self.assertRaises(HTTPException) as cm:
            service.download_file("test/protected.txt", "/tmp/download.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to download file from S3", cm.exception.detail)


    @patch('services.storage_service.boto3.client')
    def test_generate_presigned_url_successful(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        expected_url = "https://test-bucket.s3.us-east-1.amazonaws.com/test/key.txt?AWSAccessKeyId=..."
        mock_s3_instance.generate_presigned_url.return_value = expected_url
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        url = service.generate_presigned_url("test/key.txt", expiration_seconds=300)

        self.assertEqual(url, expected_url)
        mock_s3_instance.generate_presigned_url.assert_called_once_with(
            'get_object',
            Params={'Bucket': "test-bucket", 'Key': "test/key.txt"},
            ExpiresIn=300
        )

    @patch('services.storage_service.boto3.client')
    def test_generate_presigned_url_client_error(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_s3_instance.generate_presigned_url.side_effect = ClientError({'Error':{}}, "GeneratePresignedUrl")
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        with self.assertRaises(HTTPException) as cm:
            service.generate_presigned_url("test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Could not generate presigned URL", cm.exception.detail)

    @patch('services.storage_service.boto3.client')
    def test_delete_file_successful(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        result = service.delete_file("test/key.txt")

        self.assertTrue(result)
        mock_s3_instance.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    @patch('services.storage_service.boto3.client')
    def test_delete_file_client_error(self, mock_boto_client_for_service):
        mock_s3_instance = MagicMock()
        mock_s3_instance.delete_object.side_effect = ClientError({'Error':{}}, "DeleteObject")
        mock_boto_client_for_service.return_value = mock_s3_instance

        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.s3_client = mock_s3_instance
            service.bucket_name = "test-bucket"
            service.region = "us-east-1"

        with self.assertRaises(HTTPException) as cm:
            service.delete_file("test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to delete file from S3", cm.exception.detail)

    # No need to mock boto3.client for get_public_url as it doesn't make API calls
    def test_get_public_url(self):
        # Temporarily disable __init__ for this test too, or ensure StorageService can be instantiated
        # without making the head_bucket call if we are only testing this simple method.
        with patch.object(StorageService, '_StorageService__init__', lambda x: None):
            service = StorageService()
            service.bucket_name = "my-test-public-bucket"

            service.region = "us-east-1"
            self.assertEqual(service.get_public_url("some/key.jpg"), "https://my-test-public-bucket.s3.amazonaws.com/some/key.jpg")

            service.region = "eu-west-2"
            self.assertEqual(service.get_public_url("another/path.png"), "https://my-test-public-bucket.s3.eu-west-2.amazonaws.com/another/path.png")

            service.region = None # Test case where region might be None
            self.assertEqual(service.get_public_url("no/region/key"), "https://my-test-public-bucket.s3.amazonaws.com/no/region/key")


if __name__ == "__main__":
    unittest.main()
