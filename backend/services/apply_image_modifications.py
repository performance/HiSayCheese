import cv2
import numpy as np
from typing import List, Dict, Optional, Any # For type hinting

# --- Individual Enhancement Functions ---

def apply_brightness_contrast(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """
    Adjusts the brightness and contrast of an image.
    :param image: Input image as a NumPy array.
    :param brightness: Float value. 0 means no change.
                       Positive values increase brightness, negative values decrease it.
                       Expected range for UI might be -1.0 to 1.0, mapped to beta -127 to 127.
    :param contrast: Float value. 1.0 means no change.
                     Values > 1 increase contrast, values < 1 decrease it.
                     Expected range for UI might be 0.0 to 2.0, mapped to alpha 0.0 to 2.0.
    :return: Image with adjusted brightness and contrast.
    """
    if image is None:
        raise ValueError("Input image cannot be None for apply_brightness_contrast")

    # Map brightness: Assuming UI provides -1 to 1, map to beta -127 to 127
    # brightness_offset = brightness * 127  # Simple linear mapping
    # A common convention for brightness is an additive factor.
    # For convertScaleAbs, beta is an additive offset.
    # If brightness is a value like 0-100 where 50 is no change, convert to -127 to 127.
    # Assuming brightness is a value from -255 to 255 for direct beta.
    # Let's use a factor based approach for brightness as well for consistency, then map to beta.
    # A common formula is: new_pixel = alpha * old_pixel + beta
    # Here, `contrast` acts as alpha. `brightness` needs to be an offset (beta).

    # If brightness parameter is -1 to 1 (0 no change):
    # Convert to beta value (e.g. if brightness = 0.1, add 0.1 * 127 to pixels)
    # Let's assume brightness is an offset value directly from -255 to 255.
    # A common way to scale brightness: if brightness is in [-1, 1], beta = brightness * 100
    beta = brightness # Assuming brightness is already scaled, e.g. -100 to 100

    # Contrast: alpha factor. If contrast is 1, no change.
    alpha = contrast

    # new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # A more manual approach to prevent oversaturation from convertScaleAbs's direct scaling:
    # Or, more simply, use a weighted sum for more control if convertScaleAbs is too harsh
    # For brightness: image = cv2.add(image, np.array([beta])) (per channel if needed, or overall)
    # For contrast: image = cv2.multiply(image, np.array([alpha]))

    # Using the standard formula: output = alpha * input + beta
    # Ensure image is not float before this operation if it's not already
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image


def apply_saturation(image: np.ndarray, saturation_factor: float) -> np.ndarray:
    """
    Adjusts the saturation of an image.
    :param image: Input image (BGR).
    :param saturation_factor: 1.0 means no change. >1 increases saturation, <1 decreases it.
    :return: Image with adjusted saturation (BGR).
    """
    if image is None:
        raise ValueError("Input image cannot be None for apply_saturation")
    if saturation_factor == 1.0:
        return image

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # Multiply S channel by saturation_factor, ensuring it's float for multiplication
    s = s.astype(np.float32)
    s = s * saturation_factor

    # Clip S channel values to [0, 255] and convert back to uint8
    s = np.clip(s, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    saturated_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return saturated_image


def apply_background_blur(image: np.ndarray, blur_radius_pixels: int) -> np.ndarray:
    """
    Applies Gaussian blur to the entire image.
    TODO: Implement actual background segmentation for true background blur.
    :param image: Input image.
    :param blur_radius_pixels: Radius for Gaussian blur. If 0 or 1, returns original.
                               This will be used to derive kernel size.
    :return: Blurred image.
    """
    if image is None:
        raise ValueError("Input image cannot be None for apply_background_blur")
    if blur_radius_pixels <= 1: # Typically blur radius needs to be > 1 for a noticeable effect
        return image

    # Kernel size must be odd
    kernel_size = 2 * blur_radius_pixels + 1

    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def apply_crop(image: np.ndarray, crop_rect: List[int]) -> np.ndarray:
    """
    Crops the image to the given rectangle.
    :param image: Input image.
    :param crop_rect: List [x, y, w, h] for the crop area.
    :return: Cropped image.
    """
    if image is None:
        raise ValueError("Input image cannot be None for apply_crop")
    if not crop_rect or len(crop_rect) != 4:
        # Or log a warning and return original image
        raise ValueError("Invalid crop_rect provided.")

    x, y, w, h = crop_rect
    img_h, img_w = image.shape[:2]

    # Basic validation, can be made more robust
    if w <= 0 or h <= 0:
        raise ValueError("Crop width and height must be positive.")
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > img_w: w = img_w - x
    if y + h > img_h: h = img_h - y

    if w <= 0 or h <= 0: # Check again after adjustments
        # This implies crop rect was entirely outside, or resulted in zero dimension
        # Consider returning original image or raising more specific error
        return image # Or raise error

    return image[y:y+h, x:x+w]


def apply_face_smoothing(image: np.ndarray, intensity: float, face_boxes: Optional[List[List[int]]] = None) -> np.ndarray:
    """
    Applies smoothing (bilateral filter) to detected face regions.
    :param image: Input image.
    :param intensity: Float factor from 0 to 1 for smoothing. If 0, returns original.
    :param face_boxes: Optional list of face bounding boxes [[x,y,w,h], ...].
                       If None or empty, original image is returned.
    :return: Image with smoothed faces.
    """
    if image is None:
        raise ValueError("Input image cannot be None for apply_face_smoothing")
    if intensity == 0.0 or not face_boxes:
        return image

    output_image = image.copy()

    for (x, y, w, h) in face_boxes:
        face_roi = output_image[y:y+h, x:x+w]
        if face_roi.size == 0: # Skip if ROI is empty
            continue

        # Determine parameters for bilateral filter based on intensity
        # d: Diameter of each pixel neighborhood.
        #    Larger values mean more pixels are included in the smoothing.
        #    Must be odd. Let's scale it up to e.g. 15 for max intensity.
        d = int(5 + 10 * intensity)
        if d % 2 == 0: d += 1 # Ensure d is odd, min 5
        if d < 5: d = 5


        # sigmaColor: Filter sigma in the color space.
        #             Larger value means farther colors within the neighborhood will be mixed together,
        #             resulting in larger areas of semi-equal color.
        sigma_color = int(10 + 65 * intensity)

        # sigmaSpace: Filter sigma in the coordinate space.
        #             Larger value means more distant pixels will influence each other as long as their colors are close enough.
        sigma_space = int(10 + 65 * intensity)

        smoothed_face = cv2.bilateralFilter(face_roi, d, sigma_color, sigma_space)
        output_image[y:y+h, x:x+w] = smoothed_face

    return output_image

# --- Main Orchestration Function ---

def apply_all_enhancements(image_path: str, enhancement_params: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Loads an image and applies a series of enhancements based on parameters.
    :param image_path: Path to the input image.
    :param enhancement_params: Dictionary of enhancement parameters.
    :return: Modified image as a NumPy array, or None if image load fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        # logger.error(f"Failed to load image from path: {image_path}")
        return None # Or raise an exception

    # Make a copy to avoid modifying the original array if it's cached or reused
    processed_image = image.copy()

    # Order of operations can matter. This is a suggested order.
    # 1. Crop (if specified) - usually done early if region of interest is much smaller
    if 'crop_rect' in enhancement_params and enhancement_params['crop_rect']:
        try:
            processed_image = apply_crop(processed_image, enhancement_params['crop_rect'])
        except ValueError as e:
            # logger.warning(f"Skipping crop due to error: {e}")
            pass # Or re-raise if crop is critical

    # 2. Brightness/Contrast
    brightness = enhancement_params.get('brightness_target', 0.0) # Default: 0 (no change)
    contrast = enhancement_params.get('contrast_target', 1.0)   # Default: 1.0 (no change)
    if brightness != 0.0 or contrast != 1.0:
        processed_image = apply_brightness_contrast(processed_image, brightness, contrast)

    # 3. Saturation
    saturation = enhancement_params.get('saturation_target', 1.0) # Default: 1.0 (no change)
    if saturation != 1.0:
        processed_image = apply_saturation(processed_image, saturation)

    # 4. Face Smoothing (if face_boxes provided)
    face_smooth_intensity = enhancement_params.get('face_smooth_intensity', 0.0)
    face_boxes = enhancement_params.get('face_boxes', None)
    if face_smooth_intensity > 0.0 and face_boxes:
        processed_image = apply_face_smoothing(processed_image, face_smooth_intensity, face_boxes)

    # 5. Background Blur (currently full image blur)
    # This should ideally be last or after face-specific ops if it's full image blur.
    # If it were true background blur with a mask, order might change.
    blur_radius = enhancement_params.get('background_blur_radius', 0)
    if blur_radius > 0:
        processed_image = apply_background_blur(processed_image, blur_radius)

    return processed_image


if __name__ == '__main__':
    # This block can be used for basic local testing of the functions.
    # Example:
    # Create a dummy black image
    # test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # cv2.imwrite("dummy_image.png", test_img)

    # params = {
    #     "brightness_target": 20,  # Example: brightness offset
    #     "contrast_target": 1.5,   # Example: contrast factor
    #     "saturation_target": 1.8, # Example: saturation factor
    #     "background_blur_radius": 3,
    #     "crop_rect": [10, 10, 80, 80], # x, y, w, h
    #     "face_smooth_intensity": 0.5,
    #     "face_boxes": [[20,20,40,40]]
    # }
    # try:
    #     # Ensure you have an image named "dummy_image.png" or provide a valid path
    #     modified_image = apply_all_enhancements("dummy_image.png", params)
    #     if modified_image is not None:
    #         cv2.imwrite("modified_dummy_image.png", modified_image)
    #         print("Processed dummy_image.png and saved as modified_dummy_image.png")
    #     else:
    #         print("Failed to process dummy_image.png")
    # except FileNotFoundError:
    #     print("Error: dummy_image.png not found. Please create it or provide a valid image path for testing.")
    # except Exception as e:
    #     print(f"An error occurred during local testing: {e}")
    pass
