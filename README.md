# HiSayCheese - Virtual AI Headshot Studio

## Overview/Vision

HiSayCheese is a virtual AI headshot studio designed to empower users to create professional-quality headshots from their existing selfies with unparalleled ease. Our vision is to provide a seamless and engaging experience that transforms simple selfies into polished, impressive headshots, offering high perceived value. We aim to differentiate ourselves through a captivating visual journey during the enhancement process, transparency in how AI is applied, a suite of free high-quality automated enhancements, and a clear path to premium features for advanced control.

## The Problem

In today's digital world, a professional headshot is essential for online profiles, job applications, and personal branding. However, traditional professional photoshoots can be expensive and time-consuming. Conversely, standard phone selfies often lack the polish and professionalism required, leading to dissatisfaction and missed opportunities. There's a clear need for an accessible solution that delivers high-quality headshots without the associated cost or hassle.

## Our Solution

HiSayCheese offers an innovative solution: users can simply upload a selfie, and our AI-driven platform will automatically enhance it into a professional-looking headshot. This includes intelligent adjustments to the background (replacement or blur), lighting enhancements, optimal framing, and even potential subtle wardrobe tweaks. The process is designed to be quick, intuitive, and yield impressive results.

## Target Audience

*   **Job Seekers:** Needing polished headshots for resumes and online job portals.
*   **LinkedIn Users:** Aiming to enhance their professional presence.
*   **Freelancers & Consultants:** Requiring professional images for websites and proposals.
*   **Students:** Preparing for internships or their first career steps.
*   **Remote Teams:** Seeking a consistent and professional look for team directories.

## Core Concepts

*   **"One-click Headshot Makeover":** This is the cornerstone of our MVP. The goal is to allow users to achieve a significantly improved, professional headshot with minimal effort – ideally, from a single selfie upload.
*   **Visual Enhancement Journey:** We want to make the AI enhancement process transparent and engaging. Users will see a real-time, animated transition: their original selfie, a "processing" stage with visual cues of the AI at work (e.g., a virtual hand appearing to make adjustments), and then the final preview. Explanations of the changes being made will accompany this journey.
*   **Free Automated Enhancement & Premium Manual Controls:** A core set of automated enhancements will be free for all users. For those desiring more granular control, premium features will unlock manual adjustments for various aspects of the headshot.
*   **Human Language Controls:** A future-forward goal is to enable users to make adjustments using natural language commands (e.g., "make the background a bit more blurry," "brighten my face slightly").

## Features Roadmap

### Phase 1: MVP - Core Experience (Current Focus)

*   **User Selfie Upload:**
    *   Drag & drop interface.
    *   Option to use device camera.
*   **AI Background Replacement/Blur:**
    *   Options: Blurred version of original background, professional office setting, neutral studio colors (e.g., grey, white, beige).
*   **Subtle Lighting & Face Enhancement:**
    *   Automated adjustments for brightness, contrast, and saturation.
    *   Basic face smoothing to reduce minor blemishes.
*   **Automated Centering Crop & Framing Adjustment:**
    *   Intelligent cropping to standard headshot framing.
*   **Virtual Hand Animation for Auto-Enhancement Journey:**
    *   Visual feedback during the automated enhancement process, showing a virtual hand "sculpting" the image.
*   **Pre-processing:**
    *   Validate image quality (resolution, sharpness) and size before processing.

### Phase 2: Advanced Features & Enhancements

*   **Improved Face Detection & Background Segmentation:**
    *   More precise cutouts for cleaner background replacement.
*   **Advanced Face Smoothing:**
    *   Texture-preserving smoothing techniques for a more natural look.
*   **Passport/ID Photo Mode (Basic Compliance):**
    *   Automatic sizing, regulation-compliant backgrounds (e.g., white, off-white), and face positioning for common documents (e.g., US Passport).
*   **Style Templates:**
    *   Pre-defined looks (e.g., "Corporate," "Friendly," "Tech Founder") that apply a set of adjustments.
*   **AI Wardrobe Touch-up:**
    *   Subtle enhancements like a virtual plain shirt or blazer overlay if the original attire is too casual.
*   **LinkedIn Optimized Mode:**
    *   Specific cropping and enhancement settings tailored for LinkedIn profile pictures.
*   **Team/Corporate Style Mode:**
    *   Allow organizations to define a consistent style for all team members' headshots.

### Phase 3: Pro Tier & Future Vision

*   **Full Manual Controls:**
    *   **Face Centering:** Crop intensity, X/Y position, rotation, face size relative to frame.
    *   **Background:** Fine-tune blur amount, choose from an expanded library of replacement options (solids, gradients, abstract, various office/nature scenes), adjust background brightness, simulate depth of field.
    *   **Lighting:** Control exposure, contrast, highlights, shadows, color temperature, and potentially simulate key/fill light adjustments.
    *   **Skin & Details:** Advanced smoothing options, teeth whitening, eye enhancement (brightening, sharpening), overall image sharpness, targeted blemish removal.
    *   **Clothing Enhancement:** Virtual wardrobe options (shirts, blazers, ties), color adjustments for existing clothing, neckline adjustments, texture enhancements.
    *   **Final Polish:** Apply style presets, color grading via LUTs, vignette, subtle film grain, final sharpness adjustments.
*   **Premium Batch Processing:**
    *   Upload multiple selfies and apply settings globally, with the option for individual tweaks.
*   **Premium Storage & Organization:**
    *   Cloud storage for processed headshots, version history, ability to save custom presets, project folders, shareable links.
*   **Advanced Passport/ID Photo Mode:**
    *   Support for various international ID and passport photo specifications.
    *   Advanced compliance checks (e.g., eye line, head size percentage).
    *   Print layout options (e.g., multiple photos on a single sheet).
    *   Potential integration with a mail-to-home print service.
*   **Human Language Interface for making adjustments.**
*   **Export Options:**
    *   Multiple formats (JPG, PNG, TIFF).
    *   Choice of resolutions.

## Potential Technical Stack

*   **Frontend:** Next.js, React, CSS (potentially for some client-side image filters/previews).
*   **AI/ML:**
    *   MediaPipe (for face detection, landmarking, potential mask overlays for background/wardrobe).
    *   OpenCV (for image manipulation, transformations).
    *   Deepface, facenet (for face analysis, potentially quality assessment).
    *   TensorFlow.js / face-api.js (for client-side AI tasks if feasible and performant).
*   **Backend:**
    *   FastAPI (Python) for the main API.
    *   SQLAlchemy with PostgreSQL or SQLite for database.
    *   Firebase (for auth, potentially user metadata/preferences).
    *   Node.js (potentially for specific microservices or AI model interaction if not directly in Python).
*   **Cloud Storage:**
    *   AWS S3 (for storing uploaded and processed image files).
*   **Considerations:** Strategic decisions on client-side vs. backend processing for different AI effects to balance performance, cost, and user experience.

## Current State of Implementation

*   The project is currently in the detailed **planning and design phase**.
*   A foundational **Next.js project structure** has been initialized.
*   A detailed project blueprint, **`docs/blueprint.md`**, has been drafted. This document outlines the core features, development phases, and initial design considerations for the "Headshot Handcrafter" concept, which is now evolving into HiSayCheese.
*   The domain **`HiSayCheese.com` is owned** and will be the primary brand for the product.

## Google Cloud Vision API Configuration

This application uses the Google Cloud Vision API for content moderation (portrait and SafeSearch detection).
To enable this functionality, you must configure authentication by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

Steps:
1. Create or use an existing Google Cloud Project.
2. Enable the Cloud Vision API for your project.
3. Create a service account.
4. Grant the 'Cloud Vision AI User' role (or a custom role with `vision.images.annotate` permission) to the service account.
5. Download the JSON key file for this service account.
6. Securely store this key file on the server where the backend application runs.
7. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable in your backend application's environment to the absolute path of this JSON key file.

Example: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"`

Ensure the backend application has this environment variable set when it starts.

### AWS S3 for Cloud Storage

The application utilizes AWS S3 for storing all uploaded original images and any images processed or enhanced through the platform. This cloud-based storage ensures scalability and reliability.

**Required Environment Variables:**

To connect to your AWS S3 bucket, the following environment variables must be set in the backend application's environment:

*   `AWS_S3_BUCKET_NAME`: The name of your S3 bucket (e.g., `my-hisaycheese-images`).
*   `AWS_S3_REGION`: The AWS region where your S3 bucket is located (e.g., `us-east-1`).
*   `AWS_ACCESS_KEY_ID`: Your AWS access key ID.
*   `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key.

These settings are loaded by `config.py`. Ensure these variables are configured correctly for the application to interact with S3. You can set these directly in your shell, or use a `.env` file management system appropriate for your deployment.

**API Responses and Pre-signed URLs:**

When images are uploaded or processed, API endpoints that return image locations (e.g., `presigned_url` in the upload response, `processed_image_path` in enhancement responses) will provide AWS S3 pre-signed URLs. These URLs grant temporary, secure access to the image files stored in the private S3 bucket.

**Local Development with S3 Mocking:**

For local development and testing, S3 interactions are typically mocked (e.g., using `moto` as implemented in the project's tests). When running the application locally with such mocks, you might still need to set the S3-related environment variables (e.g., `AWS_S3_BUCKET_NAME`, `AWS_S3_REGION`, and dummy credentials like `AWS_ACCESS_KEY_ID="testing"`, `AWS_SECRET_ACCESS_KEY="testing"`) for the `StorageService` to initialize correctly, even though `moto` will intercept the actual S3 calls. Refer to `config.py` for how these are loaded and for any default fallback values (though for credentials, real or mock values via environment variables are expected for S3 functionality).

### AWS SES for Email Sending

The application uses AWS Simple Email Service (SES) for sending transactional emails, such as account verification emails.

**Required Environment Variables:**

*   `AWS_SES_REGION`: The AWS region where your SES service is configured (e.g., `us-east-1`). This can often be the same as your S3 region.
*   `AWS_SES_SENDER_EMAIL`: The email address that will appear as the sender (e.g., `noreply@hisaycheese.com`). **This email address must be verified in your AWS SES console for the specified region.**
*   `FRONTEND_URL`: The base URL of your frontend application (e.g., `http://localhost:3000` or `https://hisaycheese.com`). This is used to generate links in emails, such as the verification link.

The existing AWS credentials (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) configured for S3 are also used by `boto3` for SES, assuming the IAM permissions are correctly set up. These settings are loaded by `config.py`.

**Local Development with SES Mocking:**

Similar to S3, SES calls are mocked using `moto` during automated tests. For local manual testing where you might not want to send real emails, ensure the application can handle the `EmailService` potentially not sending emails (e.g., if credentials aren't fully set up or the service is conditionally disabled). The `FRONTEND_URL` should be set appropriately in your local environment if you intend to click generated links that point to your local frontend instance.

## User Account Verification

New users registering with HiSayCheese will need to verify their email address to fully activate their account and access all features.

1.  **Registration:** Upon successful registration, the system automatically sends a verification email to the address provided. This email is sent asynchronously to ensure a fast response from the registration API endpoint.
2.  **Verification Email:** The email contains a unique, time-sensitive verification link.
3.  **Verification Process:** Clicking this link directs the user to an API endpoint (`GET /api/auth/verify-email?token=<token>`) that validates the token.
4.  **Access Granted:** If the token is valid and not expired, the user's email is marked as verified in the database. They can then log in and access all protected API resources and application features. If the token is invalid or expired, an appropriate error message is displayed.

Until the email is verified, access to certain protected API endpoints will be restricted.

## Project Naming

*   **Primary Name:** HiSayCheese (utilizing the owned domain `HiSayCheese.com`)
*   **Internal/Alternative Name:** Headshot Handcrafter (as used in `docs/blueprint.md` and internal planning documents)

## How to Contribute (Placeholder)

We welcome contributions! Please see `CONTRIBUTING.md` (link to be created) for guidelines on how to get involved with the project.

## License (Placeholder)

To be determined.
