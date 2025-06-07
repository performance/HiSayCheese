import logging
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from fastapi import HTTPException, status
from typing import List, Optional

# Import configurations from config.py
try:
    from config import AWS_SES_REGION, AWS_SES_SENDER_EMAIL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
except ImportError:
    # Fallback for simpler testing or if config structure changes
    AWS_SES_REGION = os.environ.get("AWS_SES_REGION", "us-east-1")
    AWS_SES_SENDER_EMAIL = os.environ.get("AWS_SES_SENDER_EMAIL", "sender@example.com")
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.region = AWS_SES_REGION
        self.sender_email = AWS_SES_SENDER_EMAIL

        try:
            self.ses_client = boto3.client(
                "ses",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=self.region
            )
            # Verify SES setup by checking if the sender email is verified.
            # This is a best-effort check during initialization.
            # Actual verification status is crucial at the time of sending.
            self._check_sender_email_verification()
            logger.info(f"Successfully initialized SES client for region: {self.region} and sender: {self.sender_email}")

        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"SES Configuration Error: Credentials not found or incomplete. {e}")
            # Raising HTTPException here might stop app startup if EmailService is initialized globally.
            # Consider a lazy init or a health check method if this is an issue.
            # For now, let it raise, as a non-functional email service from startup is critical.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"SES credentials not configured: {e}")
        except ClientError as e:
            logger.error(f"SES ClientError during initialization: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"SES client initialization error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing SES client: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error initializing SES client: {e}")

    def _check_sender_email_verification(self):
        """
        Helper to check if the sender email is verified in SES.
        Logs a warning if not verified, as sending will fail.
        """
        try:
            response = self.ses_client.get_identity_verification_attributes(
                Identities=[self.sender_email]
            )
            attrs = response.get('VerificationAttributes', {}).get(self.sender_email)
            if not attrs or attrs.get('VerificationStatus') != 'Success':
                logger.warning(
                    f"Sender email '{self.sender_email}' is not verified in AWS SES region '{self.region}'. "
                    "Email sending will likely fail."
                )
            else:
                logger.info(f"Sender email '{self.sender_email}' is verified in AWS SES region '{self.region}'.")
        except ClientError as e:
            logger.error(f"Could not get verification status for {self.sender_email}: {e}. Assuming not verified.")
            # This might happen due to lack of permissions for GetIdentityVerificationAttributes

    def send_email(self, to_address: str, subject: str, html_body: str, text_body: Optional[str] = None):
        """
        Sends an email using AWS SES.
        :param to_address: The recipient's email address.
        :param subject: The subject of the email.
        :param html_body: The HTML content of the email.
        :param text_body: Optional plain text content for email clients that don't support HTML.
        :raises HTTPException: If email sending fails.
        """
        if not self.ses_client: # Should not happen if constructor succeeded
            logger.error("SES client not initialized. Cannot send email.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Email service not available.")

        destination = {'ToAddresses': [to_address]}

        message_body = {}
        if html_body:
            message_body['Html'] = {'Data': html_body, 'Charset': 'UTF-8'}
        if text_body: # If plain text body is also provided
             message_body['Text'] = {'Data': text_body, 'Charset': 'UTF-8'}

        if not message_body:
            logger.error("Email body (HTML or Text) is required.")
            # This should ideally be caught before calling this low-level send
            raise ValueError("HTML body or text body must be provided for the email.")


        message = {
            'Subject': {'Data': subject, 'Charset': 'UTF-8'},
            'Body': message_body
        }

        try:
            response = self.ses_client.send_email(
                Source=self.sender_email,
                Destination=destination,
                Message=message
            )
            logger.info(f"Email sent successfully to {to_address}. Message ID: {response.get('MessageId')}")
            return response.get('MessageId')
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            logger.error(f"Failed to send email to {to_address} via SES. Error: {e.response}")
            if error_code == 'MessageRejected':
                # This can happen if the recipient address is invalid, on a suppression list,
                # or if the sender email is not verified (though we check at init).
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Email was rejected by SES. Ensure sender email is verified and recipient is valid.")
            elif error_code in ['InvalidParameterValue', 'MissingParameter']:
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid parameters for SES send_email: {error_code}")
            else:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to send email via SES: {error_code or 'Unknown SES ClientError'}")
        except Exception as e:
            logger.error(f"Unexpected error sending email to {to_address}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error sending email: {e}")

    def get_verification_email_template(self, username: str, verification_url: str) -> dict:
        """
        Generates the subject and HTML body for an account verification email.
        """
        subject = "Verify your account for Our Awesome App" # Replace with actual app name

        html_body = f"""
        <html>
            <head></head>
            <body>
                <h1>Hi {username},</h1>
                <p>Thanks for signing up for Our Awesome App! Please verify your email address by clicking the link below:</p>
                <p><a href="{verification_url}">Verify Email Address</a></p>
                <p>If you did not sign up for this account, you can ignore this email.</p>
                <p>Thanks,<br>The Awesome App Team</p>
            </body>
        </html>
        """
        text_body = f"""
        Hi {username},

        Thanks for signing up for Our Awesome App! Please verify your email address by visiting the following URL:
        {verification_url}

        If you did not sign up for this account, you can ignore this email.

        Thanks,
        The Awesome App Team
        """
        return {"subject": subject, "html_body": html_body, "text_body": text_body}

# Example usage (for direct testing of this file)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Attempting to initialize EmailService...")

    # This will only work if AWS credentials and SES are configured correctly
    # and the sender email is verified in SES.
    try:
        email_service = EmailService()
        logger.info("EmailService initialized.")

        # --- Example: Send a test verification email ---
        # Note: Replace 'recipient@example.com' with a real test email address you have access to.
        # Ensure AWS_SES_SENDER_EMAIL is verified in your SES console.
        # test_recipient_email = "recipient@example.com"
        # if test_recipient_email == "recipient@example.com":
        #    print("\nWARNING: Update 'test_recipient_email' in the __main__ block of email_service.py to a real email address to test sending.\n")
        # else:
        #    print(f"Attempting to send a test verification email to: {test_recipient_email}")
        #    email_content = email_service.get_verification_email_template(
        #        username="Test User",
        #        verification_url="http://localhost:3000/verify-email?token=dummytesttoken123"
        #    )
        #    email_service.send_email(
        #        to_address=test_recipient_email,
        #        subject=email_content["subject"],
        #        html_body=email_content["html_body"],
        #        text_body=email_content["text_body"]
        #    )
        #    print(f"Test email supposedly sent to {test_recipient_email}. Check their inbox (and spam folder).")

    except HTTPException as e:
        print(f"HTTPException during EmailService test: {e.detail} (Status: {e.status_code})")
    except Exception as e:
        print(f"Generic error during EmailService test: {e}")
