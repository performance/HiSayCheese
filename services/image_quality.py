import cv2
import numpy as np
import os

def analyze_image_quality(image_path: str) -> dict:
    """
    Analyzes the brightness and contrast of an image.

    Args:
        image_path: Path to the image file.

    Returns:
        A dictionary containing:
            - brightness: Mean pixel intensity (0-1).
            - contrast: Standard deviation of pixel intensities.
            - insights: A list of strings describing the image quality (e.g., "image is dark").
    Raises:
        FileNotFoundError: If the image_path does not exist.
        ValueError: If the image cannot be loaded by OpenCV.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}. Check if the file is a valid image.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness: Mean pixel intensity
    # Normalize to 0-1 range (assuming 8-bit image)
    brightness = np.mean(gray_image) / 255.0

    # Contrast: Standard deviation of pixel intensities
    contrast = np.std(gray_image)

    insights = []
    # Define thresholds (these might need tuning)
    # Brightness thresholds (0-1 scale)
    DARK_THRESHOLD = 0.3
    BRIGHT_THRESHOLD = 0.7
    # Contrast thresholds (pixel intensity std dev)
    LOW_CONTRAST_THRESHOLD = 20
    HIGH_CONTRAST_THRESHOLD = 60 # Example, may need adjustment

    if brightness < DARK_THRESHOLD:
        insights.append("image is dark")
    elif brightness > BRIGHT_THRESHOLD:
        insights.append("image is bright")
    else:
        insights.append("image has balanced brightness") # Added for completeness

    if contrast < LOW_CONTRAST_THRESHOLD:
        insights.append("image has low contrast")
    elif contrast > HIGH_CONTRAST_THRESHOLD:
        insights.append("image has high contrast")
    else:
        insights.append("image has balanced contrast") # Added for completeness

    return {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 2),
        "insights": insights,
    }

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # This part requires creating dummy image files or using actual test images.
    # For simplicity, we'll just print a message.
    print("Image quality service module. Run tests for example usage.")
    # To test this properly, you would need to:
    # 1. Create a dummy image (e.g., using numpy and cv2.imwrite)
    #    - A very dark image (e.g., all zeros)
    #    - A very bright image (e.g., all 255s)
    #    - A low contrast image (e.g., all 128s)
    #    - A normal image
    # 2. Call analyze_image_quality with the path to the dummy image
    # 3. Print the results
    #
    # Example dummy image creation (conceptual):
    # dark_img = np.zeros((100, 100), dtype=np.uint8)
    # cv2.imwrite("dark_test_image.png", dark_img)
    # results = analyze_image_quality("dark_test_image.png")
    # print(f"Dark image results: {results}")
    # os.remove("dark_test_image.png") # Clean up
