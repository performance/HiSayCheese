import unittest
from unittest.mock import patch, MagicMock, call
import os

# Set environment variables for moto BEFORE importing EmailService or config
os.environ["AWS_SES_REGION"] = "eu-west-1" # Use a different region for SES tests
os.environ["AWS_SES_SENDER_EMAIL"] = "testsender@example.com"
os.environ["AWS_ACCESS_KEY_ID"] = "testing_ses" # For moto
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing_ses" # For moto
os.environ["AWS_SESSION_TOKEN"] = "testing_ses" # For moto

from fastapi import HTTPException, status
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Now import the service
from services.email_service import EmailService

class TestEmailService(unittest.TestCase):

    @patch('services.email_service.boto3.client') # Patch where boto3.client is used in email_service
    def test_init_successful_email_verified(self, mock_boto_client):
        mock_ses = MagicMock()
        mock_boto_client.return_value = mock_ses

        # Mock response for get_identity_verification_attributes
        mock_ses.get_identity_verification_attributes.return_value = {
            'VerificationAttributes': {
                'testsender@example.com': {'VerificationStatus': 'Success'}
            }
        }

        service = EmailService()

        mock_boto_client.assert_called_once_with(
            "ses",
            aws_access_key_id="testing_ses",
            aws_secret_access_key="testing_ses",
            region_name="eu-west-1"
        )
        mock_ses.get_identity_verification_attributes.assert_called_once_with(
            Identities=['testsender@example.com']
        )
        self.assertEqual(service.region, "eu-west-1")
        self.assertEqual(service.sender_email, "testsender@example.com")

    @patch('services.email_service.boto3.client')
    def test_init_successful_email_not_verified(self, mock_boto_client):
        mock_ses = MagicMock()
        mock_boto_client.return_value = mock_ses
        mock_ses.get_identity_verification_attributes.return_value = {
            'VerificationAttributes': {
                'testsender@example.com': {'VerificationStatus': 'Pending'}
            }
        }
        # Expect a log warning, but not an exception from __init__ itself for this case
        with self.assertLogs(logger='services.email_service', level='WARNING') as cm:
            EmailService()
        self.assertTrue(any("not verified in AWS SES" in message for message in cm.output))

    @patch('services.email_service.boto3.client')
    def test_init_no_credentials_error_on_boto_client(self, mock_boto_client):
        mock_boto_client.side_effect = NoCredentialsError()

        with self.assertRaises(HTTPException) as cm:
            EmailService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("SES credentials not configured", cm.exception.detail)

    @patch('services.email_service.boto3.client')
    def test_init_partial_credentials_error_on_boto_client(self, mock_boto_client):
        mock_boto_client.side_effect = PartialCredentialsError(provider="aws", cred_var="secret_key")
        with self.assertRaises(HTTPException) as cm:
            EmailService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("SES credentials not configured", cm.exception.detail)

    @patch('services.email_service.boto3.client')
    def test_init_client_error_on_boto_client(self, mock_boto_client):
        # This simulates ClientError directly from boto3.client(), not from get_identity_verification_attributes
        mock_boto_client.side_effect = ClientError({'Error': {'Code': 'InvalidClientTokenId'}}, "AssumeRole")
        with self.assertRaises(HTTPException) as cm:
            EmailService()
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("SES client initialization error", cm.exception.detail)

    @patch('services.email_service.boto3.client')
    def test_init_client_error_on_get_identity_verification(self, mock_boto_client):
        mock_ses = MagicMock()
        mock_boto_client.return_value = mock_ses
        mock_ses.get_identity_verification_attributes.side_effect = ClientError({'Error': {}}, "GetIdentityVerificationAttributes")
        # Expect a log error, but __init__ should still complete if client was made.
        # The _check_sender_email_verification catches this and logs.
        with self.assertLogs(logger='services.email_service', level='ERROR') as cm_log:
            EmailService()
        self.assertTrue(any("Could not get verification status" in message for message in cm_log.output))


    def _get_mocked_service_with_verified_sender(self):
        """Helper to get a service instance with a mocked SES client and verified sender."""
        self.mock_ses_client = MagicMock()
        self.mock_ses_client.get_identity_verification_attributes.return_value = {
            'VerificationAttributes': { 'testsender@example.com': {'VerificationStatus': 'Success'} }
        }
        with patch('services.email_service.boto3.client', return_value=self.mock_ses_client):
            service = EmailService()
        return service # service.ses_client will be self.mock_ses_client

    def test_send_email_successful(self):
        service = self._get_mocked_service_with_verified_sender()
        self.mock_ses_client.send_email.return_value = {'MessageId': 'test-message-id'}

        to = "recipient@example.com"
        subject = "Test Subject"
        html_body = "<p>Test HTML</p>"
        text_body = "Test Text"

        msg_id = service.send_email(to, subject, html_body, text_body)

        self.assertEqual(msg_id, "test-message-id")
        self.mock_ses_client.send_email.assert_called_once_with(
            Source="testsender@example.com",
            Destination={'ToAddresses': [to]},
            Message={
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {
                    'Html': {'Data': html_body, 'Charset': 'UTF-8'},
                    'Text': {'Data': text_body, 'Charset': 'UTF-8'}
                }
            }
        )

    def test_send_email_message_rejected(self):
        service = self._get_mocked_service_with_verified_sender()
        self.mock_ses_client.send_email.side_effect = ClientError(
            {'Error': {'Code': 'MessageRejected', 'Message': 'Email address is not verified.'}},
            "SendEmail"
        )
        with self.assertRaises(HTTPException) as cm:
            service.send_email("recipient@example.com", "Subject", "<p>Hi</p>")
        self.assertEqual(cm.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Email was rejected by SES", cm.exception.detail)

    def test_send_email_invalid_parameter(self):
        service = self._get_mocked_service_with_verified_sender()
        self.mock_ses_client.send_email.side_effect = ClientError(
            {'Error': {'Code': 'InvalidParameterValue', 'Message': 'Invalid to address'}},
            "SendEmail"
        )
        with self.assertRaises(HTTPException) as cm:
            service.send_email("invalid-email", "Subject", "<p>Hi</p>")
        self.assertEqual(cm.exception.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Invalid parameters for SES send_email", cm.exception.detail)

    def test_send_email_other_client_error(self):
        service = self._get_mocked_service_with_verified_sender()
        self.mock_ses_client.send_email.side_effect = ClientError(
            {'Error': {'Code': 'InternalFailure', 'Message': 'SES down'}},
            "SendEmail"
        )
        with self.assertRaises(HTTPException) as cm:
            service.send_email("recipient@example.com", "Subject", "<p>Hi</p>")
        self.assertEqual(cm.exception.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Failed to send email via SES: InternalFailure", cm.exception.detail)

    @patch('services.email_service.boto3.client') # Add patch to allow instantiation
    def test_get_verification_email_template(self, mock_boto_client_ignored):
        # Mock the get_identity_verification_attributes call within __init__
        mock_ses_init_instance = MagicMock()
        mock_ses_init_instance.get_identity_verification_attributes.return_value = {
            'VerificationAttributes': { 'testsender@example.com': {'VerificationStatus': 'Success'} }
        }
        mock_boto_client_ignored.return_value = mock_ses_init_instance

        service = EmailService()

        username = "JohnDoe"
        verification_url = "http://app.com/verify?token=mytoken123"

        email_content = service.get_verification_email_template(username, verification_url)

        self.assertIsInstance(email_content, dict)
        self.assertIn("subject", email_content)
        self.assertIn("html_body", email_content)
        self.assertIn("text_body", email_content)

        self.assertIn(username, email_content["html_body"])
        self.assertIn(verification_url, email_content["html_body"])
        self.assertIn(username, email_content["text_body"])
        self.assertIn(verification_url, email_content["text_body"])
        self.assertIn("Our Awesome App", email_content["subject"]) # Check app name placeholder

if __name__ == "__main__":
    unittest.main()
