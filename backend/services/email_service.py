# services/email_service.py

import logging
from typing import Optional
from fastapi import Request, HTTPException, status

# Import configurations from config.py
# We will add EMAIL_SERVICE_TYPE to config.py
from config import (
    EMAIL_SERVICE_TYPE, 
    SENDER_EMAIL_ADDRESS,
    AWS_SES_REGION, 
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY
)

logger = logging.getLogger(__name__)

class EmailService:
    """
    An email service that can operate in two modes:
    - 'console': Prints email content to the console. For local development.
    - 'ses': Sends emails using AWS SES. For production.
    
    The mode is determined by the `EMAIL_SERVICE_TYPE` environment variable.
    """
    def __init__(self, request: Optional[Request] = None):
        self.request = request
        self.mode = EMAIL_SERVICE_TYPE
        self.sender_email = SENDER_EMAIL_ADDRESS

        logger.info(f"Initializing EmailService in '{self.mode}' mode.")
        
        if self.mode == "console":
            self.client = self._get_console_client()
        elif self.mode == "ses":
            self.client = self._get_ses_client()
        else:
            raise ValueError(f"Invalid EMAIL_SERVICE_TYPE: '{self.mode}'. Must be 'console' or 'ses'.")
    
    def _get_console_client(self):
        """Returns a mock client for console-only logging."""
        logger.info("Email client is in 'console' mode. Emails will be printed to the terminal.")
        return ConsoleEmailClient(sender_email=self.sender_email)

    def _get_ses_client(self):
        """Initializes and returns the AWS SES client."""
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, self.sender_email]):
                logger.error("SES mode is active, but required AWS credentials or sender email are missing.")
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Email service is not configured.")

            client = boto3.client(
                'ses',
                region_name=AWS_SES_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"AWS SES client initialized for region '{AWS_SES_REGION}'.")
            # Note: We skip the noisy verification check on startup.
            return client
        except ImportError:
            logger.error("Boto3 is not installed, but EMAIL_SERVICE_TYPE is 'ses'.")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Email service dependencies are not installed.")
        except (NoCredentialsError, ClientError) as e:
            logger.error(f"Failed to initialize AWS SES client: {e}")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Could not connect to email service: {e}")

    def send_email(self, to_address: str, subject: str, html_body: str, text_body: Optional[str] = None):
        """Sends an email using the configured client (Console or SES)."""
        if self.mode == 'console':
            # The console client has its own 'send_email' method
            return self.client.send_email(to_address, subject, html_body, text_body)
        
        # --- SES Sending Logic ---
        try:
            message_body = {}
            if html_body:
                message_body['Html'] = {'Data': html_body, 'Charset': 'UTF-8'}
            if text_body:
                message_body['Text'] = {'Data': text_body, 'Charset': 'UTF-8'}

            if not message_body:
                raise ValueError("HTML or text body must be provided.")

            response = self.client.send_email(
                Source=self.sender_email,
                Destination={'ToAddresses': [to_address]},
                Message={
                    'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                    'Body': message_body
                }
            )
            logger.info(f"Email sent successfully to {to_address} via SES. Message ID: {response.get('MessageId')}")
            return True
        except Exception as e:
            # Using the actual Boto3 exception type if available
            try:
                from botocore.exceptions import ClientError
                if isinstance(e, ClientError):
                    logger.error(f"Failed to send email to {to_address} via SES: {e.response['Error']['Message']}")
                    # Don't crash the app, just log the failure to send.
                    return False
            except ImportError:
                pass
            
            logger.error(f"An unexpected error occurred sending email to {to_address}: {e}")
            return False

    def get_verification_email_template(self, username: str, verification_url: str) -> dict:
        """Generates the content for an account verification email."""
        # ... (This function remains the same as your original)
        subject = "Verify your account for Our Awesome App"
        html_body = f"""...""" # Keep your HTML body here
        text_body = f"""...""" # Keep your text body here
        return {"subject": subject, "html_body": html_body, "text_body": text_body}

# --- Helper Class for Console Mode ---
class ConsoleEmailClient:
    """A mock email client that prints emails to the console."""
    def __init__(self, sender_email: str):
        self.sender_email = sender_email

    def send_email(self, to_address: str, subject: str, html_body: str, text_body: Optional[str] = None):
        print("\n" + "="*80)
        print("--- [Email Sent (Console Mode)] ---")
        print(f"  From: {self.sender_email}")
        print(f"  To: {to_address}")
        print(f"  Subject: {subject}")
        print("--- [HTML Body] ---")
        print(html_body.strip())
        print("="*80 + "\n")
        logger.info(f"Printed email to console for recipient: {to_address}")
        return True # Simulate successful sending