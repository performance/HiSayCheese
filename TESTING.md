# Testing in HiSayCheese

## Introduction & Testing Philosophy

Testing is a critical part of the HiSayCheese development process. Our goal is to maintain a robust and well-tested codebase. This ensures the reliability of new features, prevents regressions, and makes the application more maintainable in the long run. We believe that a strong testing culture empowers developers to contribute with confidence.

## Types of Tests

We employ several types of tests to ensure application quality:

### Backend Testing

Our backend, built with FastAPI, utilizes `pytest` for its testing framework.

*   **Unit Tests**: These tests focus on individual functions, classes, or modules in isolation to ensure they behave as expected. They are crucial for verifying the core logic of our application components.
*   **Integration Tests**: These tests verify the interactions between different parts of the backend system. This includes testing API endpoints, database interactions (potentially with services like `UserService` or `ImageService`), and integrations with external services. For AWS services, we use `moto` to mock interactions, allowing us to test service integrations without actual calls to AWS.

**Location**: All backend tests can be found in the `backend/tests/` directory.

### Frontend Testing

Our frontend is built with Next.js and React.

*   **Unit Tests / Component Tests**: These tests focus on individual React components, hooks, or utility functions. The goal is to ensure each UI piece and its logic functions correctly in isolation. We recommend using tools like **Jest** and **React Testing Library** for this. The file `frontend/src/lib/test-utils.ts` may contain helpful utilities for testing.
*   **End-to-End (E2E) Tests (Future Goal)**: While not yet implemented, we aim to introduce E2E tests in the future. These tests will simulate real user scenarios by interacting with the application through the UI, covering complete user flows from start to finish. Tools like Cypress or Playwright will be considered for this purpose.

**Location**: Frontend tests can be colocated with the components they test (e.g., `frontend/src/components/MyComponent.test.tsx`) or reside in a dedicated test directory like `frontend/src/__tests__/`.

## How to Run Tests

For detailed instructions on setting up your environment and running tests, please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file. Below is a summary of the commands:

### Backend Tests

1.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
2.  Ensure your Python virtual environment is activated.
3.  Run all tests using pytest:
    ```bash
    pytest
    ```

### Frontend Tests

1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Run the test script (if configured):
    ```bash
    npm test
    # Or using yarn:
    # yarn test
    ```
    **Note**: As mentioned in `CONTRIBUTING.md`, an explicit `test` script might not yet be configured in `frontend/package.json`. Frontend testing setup (e.g., with Jest/React Testing Library) is an area for future improvement. You can use `npm run typecheck` for static type checking.

## Testing Tools & Libraries

### Backend

*   **pytest**: The primary framework for writing and running all types of backend tests.
*   **moto**: Used for mocking AWS services during integration tests, ensuring that our tests for services interacting with S3, etc., are reliable and don't depend on live AWS resources.
*   **httpx**: Often used with `TestClient` from FastAPI for testing API endpoints.

### Frontend (Recommended)

*   **Jest**: A popular JavaScript testing framework.
*   **React Testing Library**: For testing React components in a way that resembles how users interact with them.
*   **TypeScript**: For static type checking, which helps catch errors early (`npm run typecheck`).

## Contribution Guidelines for Testing

We encourage all contributors to include tests with their code.

*   **New Features**: Any new feature should be accompanied by tests that cover its functionality. This includes unit tests for new logic and integration tests if the feature involves multiple components or services.
*   **Bug Fixes**: When fixing a bug, it is highly recommended to first write a test that reproduces the bug. This test should fail before the fix and pass after the fix is applied. This helps prevent regressions.
*   **Run Tests Locally**: Before submitting a pull request, please ensure all backend and relevant frontend tests pass in your local environment.
*   **Improve Coverage**: If you identify areas of the codebase that are not well-tested, contributions that improve test coverage are always welcome.

By following these guidelines, we can collectively ensure HiSayCheese remains a high-quality and dependable application.
```
