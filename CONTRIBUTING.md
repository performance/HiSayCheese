# Contributing to HiSayCheese

First off, thank you for considering contributing to HiSayCheese! It's people like you that make HiSayCheese such a great tool. We welcome contributions of all kinds, from bug fixes to new features and documentation improvements.

## Table of Contents

-   [Development Setup](#development-setup)
    -   [Prerequisites](#prerequisites)
    -   [Cloning the Repository](#cloning-the-repository)
    -   [Backend Setup](#backend-setup)
    -   [Frontend Setup](#frontend-setup)
    -   [Environment Variables](#environment-variables)
-   [Running the Application](#running-the-application)
    -   [Running the Backend](#running-the-backend)
    -   [Running the Frontend](#running-the-frontend)
-   [Linting and Formatting](#linting-and-formatting)
    -   [Backend Linting/Formatting](#backend-lintingformatting)
    -   [Frontend Linting/Formatting](#frontend-lintingformatting)
-   [Running Tests](#running-tests)
    -   [Backend Tests](#backend-tests)
    -   [Frontend Tests](#frontend-tests)
-   [Submitting Pull Requests](#submitting-pull-requests)
    -   [Branching Strategy](#branching-strategy)
    -   [Commit Messages](#commit-messages)
    -   [PR Checklist](#pr-checklist)

## Development Setup

### Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **Node.js:** We recommend using the latest LTS version. You can download it from [nodejs.org](https://nodejs.org/).
*   **npm or Yarn:** npm is included with Node.js. Yarn can be installed from [yarnpkg.com](https://yarnpkg.com/). This guide will use `npm` in examples. The project appears to use npm based on `package-lock.json`, but `package.json` specifies yarn under `packageManager`. Examples will use npm.
*   **Python:** We recommend using Python 3.11.9. You can download it from [python.org](https://www.python.org/). or use pyenv.
*   **pip:** Python's package installer, usually comes with Python.

### Cloning the Repository

1.  Fork the repository on GitHub.
2.  Clone your forked repository to your local machine:

    ```bash
    git clone https://github.com/YOUR_USERNAME/HiSayCheese.git
    cd HiSayCheese
    ```

### Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
### Backend organization 

models/ directory: This should only contain SQLAlchemy models—the Python classes that map directly to your database tables. They define the structure of your data at rest.
schemas/ directory: This should only contain Pydantic models—the Python classes that define the shape of your API requests and responses. They define the structure of your data in motion.
crud.py: This should contain functions that interact with the database (Session). They act as a bridge, taking data (often from Pydantic schemas) and using it to create, read, update, or delete SQLAlchemy model instances.
routers/ directory: This should contain the API endpoint logic. It handles HTTP requests, calls services or CRUD functions, and returns HTTP responses.

#### Backend smoke test .. 

``` bash
→curl -X POST "http://127.0.0.1:8000/api/images/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_one.png"

```
expect a response like .. 
``` bash   
{"filename":"test_one.png","filepath":"original_images/17c51a86abbb41f3ac3f079ce7b5b0ef.png","filesize":2359439,"mimetype":"image/png","width":1330,"height":1318,"format":"PNG","exif_orientation":null,"color_profile":"ICC","rejection_reason":null,"id":"ac32fe1e-c396-4d6a-8d00-394b4a9bbd9f","created_at":"2025-06-11T04:22:54.649825","presigned_url":"http://127.0.0.1:8000/static/original_images/17c51a86abbb41f3ac3f079ce7b5b0ef.png"}%    
```

If you get it, **Congratulations!** 🎉

That JSON response is the beautiful result of all the refactoring and bug fixing. It signifies that your core image upload workflow is now **fully functional** from end to end.

Let's break down what that successful response means:

1.  **Request Received:** Your `curl` command correctly sent the `test_one.png` file to the FastAPI server.
2.  **Middleware Executed:** All your middleware (Request ID, Logging, CORS, Rate Limiting) ran without errors.
3.  **Rate Limiter Worked:** The `slowapi` rate limiter was invoked, it correctly identified you (likely by IP address), applied the dynamic limit, and allowed the request to proceed.
4.  **Endpoint Logic Triggered:** The `upload_image` function in `routers/images.py` was called.
5.  **Validation Passed:**
    *   File size was checked and was under the 15MB limit.
    *   `python-magic` identified the file as `image/png`, which is an allowed MIME type.
6.  **Services Ran Successfully:**
    *   The (mock) malware scan passed.
    *   The **mock content moderation** service (the "local" mode we just implemented) instantly approved the image, correctly bypassing the failing Google Vision API call.
    *   Metadata (width, height, etc.) was extracted using Pillow.
7.  **Database Interaction Succeeded:**
    *   The `crud.create_image` function was called with a correctly populated `ImageCreate` schema.
    *   It successfully created a new `Image` record in your SQLite database. The `id` and `created_at` fields were generated by the DB/SQLAlchemy.
8.  **Local Storage Worked:**
    *   The `StorageService` (in "local" mode) saved the `test_one.png` file to `./local_storage/original_images/` with a new unique UUID-based filename.
9.  **Response Generated:**
    *   The final database object was converted back to a Pydantic `ImageSchema`.
    *   The `StorageService` generated a local URL (`http://127.0.0.1:8000/static/...`) for the `presigned_url` field.
    *   The complete JSON response was sent back to your `curl` client with a `201 Created` status (implied by the successful response body).

    You Can Verify the Result!

    1.  **Check the file:** Go to your `backend` directory. You should now see a new folder named `local_storage`. Inside it will be `original_images`, and inside that, you'll find the newly saved PNG file (`17c51a86abbb41f3ac3f079ce7b5b0ef.png`).
    2.  **Check the URL:** Copy the `presigned_url` from the JSON response and paste it into your web browser. The image should load perfectly!
    3.  **Check the database:** You can use a tool like `DB Browser for SQLite` to open your `database.db` file and see the new row in the `images` table.



### Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd ../frontend # Assuming you are in the backend directory
    # Or from the root: cd frontend
    ```
2.  Install the JavaScript dependencies:
    ```bash
    npm install
    # Or if you prefer yarn 
    # yarn install
    ```

### Environment Variables

Both the frontend and backend applications require environment variables for configuration.

*   **Backend:**
    *   Navigate to the `backend` directory.
    *   Create a `.env` file. Since a `.env.example` file is not provided, you will need to define the variables based on the requirements outlined in `backend/config.py` or other backend configuration files.
    *   Edit the `.env` file with your local configuration (e.g., database connection strings, AWS credentials for local development/mocking, `GOOGLE_APPLICATION_CREDENTIALS`). For local development, you might use mock services or local instances.
*   **Frontend:**
    *   Navigate to the `frontend` directory.
    *   Create a `.env.local` file. Since a `.env.example` or similar is not provided, you will need to define variables such as `NEXT_PUBLIC_API_BASE_URL`. For example:
        ```
        NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api
        ```
    *   Refer to `frontend/config/next.config.js`, `frontend/config/next.config.ts`, or relevant files for how environment variables are used.

**Important:** `.env` and `.env.local` files should not be committed to version control. They are typically listed in `.gitignore`.

## Running the Application

### Running the Backend

1.  Ensure your Python virtual environment is activated and you are in the `backend` directory.
2.  Start the FastAPI development server:
    ```bash
    uvicorn main:app --reload --port 8000
    ```
    The backend API should now be accessible at `http://localhost:8000`.

### Running the Frontend

1.  Navigate to the `frontend` directory.
2.  Start the Next.js development server:
    ```bash
    npm run dev
    ```
    The frontend application should now be accessible at `http://localhost:9002` (as specified in `package.json`).

## Linting and Formatting

### Backend Linting/Formatting

Our Python backend uses tools like Flake8 for linting and Black for formatting.

1.  Ensure your Python virtual environment is activated and you are in the `backend` directory.
2.  You may need to install these tools if they are not already part of your environment:
    ```bash
    pip install flake8 black
    ```
3.  To check for linting issues:
    ```bash
    flake8 .
    ```
4.  To automatically format the code:
    ```bash
    black .
    ```
    It's good practice to run these before committing your changes. Some projects might have pre-commit hooks configured to do this automatically.

### Frontend Linting/Formatting

Our frontend uses tools like ESLint for linting and Prettier for formatting.

1.  Navigate to the `frontend` directory.
2.  To check for linting issues (script available in `package.json`):
    ```bash
    npm run lint
    ```
3.  To automatically format the code, `package.json` does not have a specific `format` script. You can run Prettier directly:
    ```bash
    npx prettier --write .
    ```
    Check `package.json` for available scripts or project documentation for specific formatting commands.

## Running Tests

### Backend Tests

The backend uses `pytest` for running tests. `pytest` is included in `backend/requirements.txt`.

1.  Ensure your Python virtual environment is activated and you are in the `backend` directory.
2.  Run all tests:
    ```bash
    pytest
    ```
    You can also run specific test files or tests:
    ```bash
    pytest tests/test_users.py
    pytest tests/test_auth.py::test_login_for_access_token
    ```

### Frontend Tests

The frontend should use a framework like Jest with React Testing Library, 
but a specific `test` script is not there yet in `package.json`.

1.  Navigate to the `frontend` directory.
2.  There is no explicit `npm test` script yet. You need to create all of it:
    *   Update project documentation for instructions on running tests.
    *   Configure a test runner like Jest or Vitest if not already set up.
    *   Use available scripts like `npm run typecheck` (for TypeScript checking) as part of the testing.

## Submitting Pull Requests

We actively welcome your pull requests!

### Branching Strategy

*   Create your feature branches from the `main` branch. 
*   Name your branches descriptively, e.g., `feature/add-new-filter` or `bugfix/fix-login-issue`.

### Commit Messages

*   Use clear and concise commit messages.
*   Follow conventional commit formats (e.g., `feat: add user profile page`, `fix: resolve issue with image upload`).
*   A good commit message should briefly explain *what* was changed and *why*.

### PR Checklist

Before submitting your PR, please ensure you have:

1.  **Forked the repository and created your branch from `main`.**
2.  **Made your changes in a separate branch.** Do not commit directly to `main` in your fork if you intend to send a PR.
3.  **Followed the coding style guides** for the project.
4.  **Run linters and formatters** and fixed any issues.
5.  **Added relevant tests** for any new features or bug fixes.
6.  **Ensured all tests pass** locally (e.g., `pytest` for backend; verify frontend testing strategy).
7.  **Updated documentation** if your changes affect user-facing features or how developers build/run the project.
8.  **Tested your changes end-to-end.** This means verifying that your changes work as expected within the entire application flow, not just in isolation. For example, if you changed a backend API, ensure the frontend that consumes it still works correctly.
9.  **Written a clear and detailed PR description.** Explain the problem you are solving and the changes you made. Include screenshots or GIFs if they help illustrate UI changes.
10.  **Linked any relevant issues** in your PR description (e.g., "Closes #123").

Once you've submitted your PR, a team member will review it. 

Thank you for contributing!
```
