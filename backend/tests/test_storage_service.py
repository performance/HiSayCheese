import unittest
from unittest.mock import patch, MagicMock, call
import os

# Environment variables for moto are now set in conftest.py's pytest_configure hook.
# os.environ["AWS_S3_BUCKET_NAME"] = "test-bucket"
# os.environ["AWS_S3_REGION"] = "us-east-1"
# os.environ["AWS_ACCESS_KEY_ID"] = "testing" # For moto
# os.environ["AWS_SECRET_ACCESS_KEY"] = "testing" # For moto
# os.environ["AWS_SESSION_TOKEN"] = "testing" # For moto, if assuming IAM role

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

    # No setUp needed if StorageService is instantiated in each test
    # and conftest.py handles global mock setup.

    def test_init_successful(self):
        # StorageService() will be called.
        # __init__ calls boto3.client('s3'), which should be mocked by conftest.mock_boto3_module.
        # The mock_s3_client from conftest has head_bucket mocked to return True.
        service = StorageService()

        # Check if attributes are set as expected from os.environ
        self.assertEqual(service.bucket_name, "test-bucket")
        self.assertEqual(service.region, "us-east-1")

        # Check that s3_client is a MagicMock (from conftest)
        self.assertIsInstance(service.s3_client, MagicMock)

        # Check that head_bucket was called during __init__
        service.s3_client.head_bucket.assert_called_with(Bucket="test-bucket")

    def test_init_s3_connection_error_no_such_bucket(self):
        # This test requires manipulating the mock behavior for head_bucket
        # The mock_s3_client is shared from conftest.py. We need to change its behavior for this test.

        # Temporarily change side_effect of head_bucket for this test
        # This assumes conftest.py's mock_s3_client is accessible or re-patching.
        # For simplicity, we'll assume that StorageService() will get the globally mocked s3_client,
        # and we change its behavior *before* instantiating StorageService for this test.

        # To do this safely without affecting other tests if they run in parallel (not an issue for unittest)
        # or if the mock is shared, it's tricky.
        # A cleaner way might be to patch 'services.storage_service.boto3.client' again just for this test
        # to return a new mock with specific side_effects, but that defeats conftest global mock.

        # Let's assume we can access and modify the conftest mock for this test's scope.
        # This requires conftest.py to provide access to mock_s3_client or make it patchable here.
        # For now, let's assume the global mock is in place and we patch its specific method's side_effect.

        # This test is problematic with a shared module-scoped mock if we want to change side_effects per test.
        # A common pattern is to make the mock available as a fixture itself.
        # For now, we will patch `boto3.client` again, specifically for this test,
        # shadowing the conftest.py one. This isn't ideal but demonstrates the test logic.
        with patch('services.storage_service.boto3.client') as mock_boto_client:
            mock_s3_specific = MagicMock()
            error_response = {'Error': {'Code': 'NoSuchBucket'}}
            mock_s3_specific.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
            mock_boto_client.return_value = mock_s3_specific

            with self.assertRaises(HTTPException) as cm:
                StorageService()
            self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            self.assertIn("S3 bucket 'test-bucket' not found or access denied", cm.exception.detail)

    # Similar adjustments would be needed for other __init__ failure tests:
    # test_init_no_credentials_error, test_init_partial_credentials_error, etc.
    # They would each need to patch 'services.storage_service.boto3.client' to set a specific side_effect
    # for the boto3.client('s3').head_bucket call for that test's StorageService() instantiation.

    def test_upload_file_successful(self):
        service = StorageService() # Uses mock from conftest
        service.s3_client.reset_mock() # Reset from __init__ calls or previous tests
        service.s3_client.upload_fileobj.return_value = None # Explicitly set successful behavior
        service.s3_client.upload_fileobj.side_effect = None # Ensure no prior side effect interferes

        file_obj = BytesIO(b"test data")
        object_key = "test/key.txt"

        result = service.upload_file(file_obj, object_key, content_type="text/plain", acl="public-read")

        self.assertEqual(result, object_key)
        service.s3_client.upload_fileobj.assert_called_once_with(
            file_obj,
            "test-bucket",
            object_key,
            ExtraArgs={'ACL': 'public-read', 'ContentType': 'text/plain'}
        )

    def test_upload_file_client_error(self):
        service = StorageService()
        service.s3_client.reset_mock()
        service.s3_client.upload_fileobj.side_effect = ClientError({'Error': {'Code': 'AccessDenied'}}, "UploadFileobj")

        file_obj = BytesIO(b"test data")
        with self.assertRaises(HTTPException) as cm:
            service.upload_file(file_obj, "test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to upload file to S3", cm.exception.detail)

    def test_download_file_successful(self):
        service = StorageService()
        service.s3_client.reset_mock()
        service.s3_client.download_file.return_value = None # Explicitly set successful behavior
        service.s3_client.download_file.side_effect = None # Ensure no prior side effect interferes

        object_key = "test/key.txt"
        destination_path = "/tmp/test_download.txt" # Test environment should handle tmp files

        service.download_file(object_key, destination_path)

        service.s3_client.download_file.assert_called_once_with(
            "test-bucket",
            object_key,
            destination_path
        )

    def test_download_file_client_error_404(self):
        service = StorageService()
        service.s3_client.reset_mock()
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}} # Corrected to string '404'
        service.s3_client.download_file.side_effect = ClientError(error_response, "DownloadFile")

        with self.assertRaises(HTTPException) as cm:
            service.download_file("test/nonexistent.txt", "/tmp/download.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("File not found in S3", cm.exception.detail)

    def test_download_file_client_error_other(self):
        service = StorageService()
        service.s3_client.reset_mock()
        error_response = {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}}
        service.s3_client.download_file.side_effect = ClientError(error_response, "DownloadFile")

        with self.assertRaises(HTTPException) as cm:
            service.download_file("test/protected.txt", "/tmp/download.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to download file from S3", cm.exception.detail)

    def test_generate_presigned_url_successful(self):
        service = StorageService()
        service.s3_client.reset_mock()

        expected_url = "http://mocked.s3.url/test-file" # Use the value from conftest or a specific one
        service.s3_client.generate_presigned_url.return_value = expected_url
        service.s3_client.generate_presigned_url.side_effect = None # Ensure no prior side effect interferes

        url = service.generate_presigned_url("test/key.txt", expiration_seconds=300)

        self.assertEqual(url, expected_url)
        service.s3_client.generate_presigned_url.assert_called_once_with(
            'get_object',
            Params={'Bucket': "test-bucket", 'Key': "test/key.txt"},
            ExpiresIn=300
        )

    def test_generate_presigned_url_client_error(self):
        service = StorageService()
        service.s3_client.reset_mock()
        service.s3_client.generate_presigned_url.side_effect = ClientError({'Error':{}}, "GeneratePresignedUrl")

        with self.assertRaises(HTTPException) as cm:
            service.generate_presigned_url("test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Could not generate presigned URL", cm.exception.detail)

    def test_delete_file_successful(self):
        service = StorageService()
        service.s3_client.reset_mock()
        service.s3_client.delete_object.return_value = None # Explicitly set successful behavior
        service.s3_client.delete_object.side_effect = None # Ensure no prior side effect interferes

        result = service.delete_file("test/key.txt")

        self.assertTrue(result)
        service.s3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    def test_delete_file_client_error(self):
        service = StorageService()
        service.s3_client.reset_mock()
        service.s3_client.delete_object.side_effect = ClientError({'Error':{}}, "DeleteObject")

        with self.assertRaises(HTTPException) as cm:
            service.delete_file("test/key.txt")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to delete file from S3", cm.exception.detail)

    def test_get_public_url(self):
        # This test does not interact with boto3.client, so no mock interaction needed.
        # It relies on bucket_name and region attributes which are set by __init__.
        service = StorageService() # __init__ runs, uses mocked boto3.client from conftest

        # bucket_name and region should be from os.environ due to config.py
        self.assertEqual(service.bucket_name, "test-bucket")
        self.assertEqual(service.region, "us-east-1") # As set by os.environ at top of file

        # Corrected expectation: us-east-1 does not include region in the hostname for path-style URLs typically
        self.assertEqual(service.get_public_url("some/key.jpg"), "https://test-bucket.s3.amazonaws.com/some/key.jpg")

        # To test other regions, we'd need to reinstantiate StorageService with different env vars,
        # or directly set service.region, but that bypasses __init__ logic for region if any.
        # For this test, verifying with the globally set region is sufficient.
        # If we needed to test different regions:
        # with patch.dict(os.environ, {"AWS_S3_REGION": "eu-west-2"}):
        #    importlib.reload(config) # Reload config to pick up new region
        #    service_eu = StorageService() # New instance
        #    self.assertEqual(service_eu.get_public_url("another/path.png"), "https://test-bucket.s3.eu-west-2.amazonaws.com/another/path.png")


if __name__ == "__main__":
    unittest.main()
