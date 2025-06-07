import cv2
import os

def detect_faces(image_path: str) -> list:
    """
    Detects faces in an image using a Haar Cascade model.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of dictionaries, where each dictionary contains the
        bounding box ('box') and confidence score ('confidence') for a detected face.
        Returns an empty list if no faces are detected or if the image cannot be loaded.
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

    # Load the Haar Cascade model for face detection.
    # This path assumes that OpenCV is installed correctly and knows where its data files are.
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

    if not os.path.exists(cascade_path):
        # This is a fallback or a place to add download logic if needed.
        # For now, we'll raise an error if the cascade file is not found.
        raise FileNotFoundError(f"Haar Cascade model not found at {cascade_path}. Please ensure OpenCV is correctly installed or provide a valid path.")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces
    faces_detected = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces_detected:
        # Haar Cascades do not provide a direct confidence score for detections.
        # We'll use a default value of 1.0 for detected faces.
        results.append({'box': [int(x), int(y), int(w), int(h)], 'confidence': 1.0})

    return results

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # Create a dummy image file for testing if you don't have one.
    # For example, you can save a simple black image.
    # This part would require an actual image to test.
    # try:
    #     # Replace with a path to a test image containing faces
    #     test_image_path = "path_to_your_test_image.jpg"
    #     if not os.path.exists(test_image_path):
    #         print(f"Test image not found at {test_image_path}, skipping example usage.")
    #     else:
    #         detected_faces = detect_faces(test_image_path)
    #         if detected_faces:
    #             print(f"Detected {len(detected_faces)} faces:")
    #             for face in detected_faces:
    #                 print(f"  Box: {face['box']}, Confidence: {face['confidence']}")
    #         else:
    #             print("No faces detected.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    pass
