import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from typing import List, Dict, Any, Optional # Added Optional
import mediapipe as mp

# Placeholder for face detection results if needed directly
# from services.face_detection import detect_faces

# It's good practice to define constants for thresholds or factors if they might change
BLUR_MASK_THRESHOLD = 0.5 # Example threshold for MediaPipe mask

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass

def load_image_pil(image_path: str) -> Image.Image:
    """Loads an image using Pillow."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    try:
        img = Image.open(image_path)
        # Ensure image is in RGB or RGBA for consistency, as some operations might fail on P mode etc.
        if img.mode == 'P': # Palette mode
            img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
        elif img.mode == 'L': # Grayscale
            img = img.convert('RGB') # Convert to RGB to allow color enhancements
        # For RGBA, many Pillow filters work directly. If issues, convert to RGB and handle alpha separately.
        # For now, let's assume RGB is generally what we want for enhancements, then re-apply alpha if needed.
        # However, ImageEnhance works on RGBA too.
        return img
    except Exception as e:
        raise ImageProcessingError(f"Error loading image {image_path} with Pillow: {e}")

def load_image_cv2(image_path: str) -> np.ndarray:
    """Loads an image using OpenCV."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    try:
        img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Load with alpha if present
        if img_cv is None:
            raise ImageProcessingError(f"Could not load image {image_path} with OpenCV. Check file format or integrity.")
        return img_cv
    except Exception as e:
        raise ImageProcessingError(f"Error loading image {image_path} with OpenCV: {e}")

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Converts a Pillow image to an OpenCV image (NumPy array)."""
    if pil_image.mode == 'RGBA':
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
    elif pil_image.mode == 'RGB':
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    elif pil_image.mode == 'L': # Grayscale
        return np.array(pil_image) # OpenCV handles grayscale directly
    elif pil_image.mode == 'P': # Palette mode, convert to RGB first
        pil_image_rgb = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image_rgb), cv2.COLOR_RGB2BGR)
    else:
        # Attempt a generic conversion, but might not be ideal for all modes
        return np.array(pil_image.convert('RGB'))


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Converts an OpenCV image (NumPy array) to a Pillow image."""
    if len(cv2_image.shape) == 2: # Grayscale
        return Image.fromarray(cv2_image, mode='L')
    elif cv2_image.shape[2] == 3: # BGR
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB), mode='RGB')
    elif cv2_image.shape[2] == 4: # BGRA
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA), mode='RGBA')
    else:
        raise ImageProcessingError(f"Unsupported OpenCV image format for Pillow conversion: shape {cv2_image.shape}")


def apply_brightness_contrast_saturation(
    pil_img: Image.Image,
    brightness_target: float,
    contrast_target: float,
    saturation_target: float
) -> Image.Image:
    """Applies brightness, contrast, and saturation using Pillow's ImageEnhance."""
    try:
        enhancer = ImageEnhance.Brightness(pil_img)
        img_enhanced = enhancer.enhance(brightness_target)

        enhancer = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enhancer.enhance(contrast_target)

        enhancer = ImageEnhance.Color(img_enhanced) # Color is for Saturation
        img_enhanced = enhancer.enhance(saturation_target)
        return img_enhanced
    except Exception as e:
        raise ImageProcessingError(f"Error applying B/C/S enhancements: {e}")

def apply_crop(pil_img: Image.Image, crop_rect: List[int]) -> Image.Image:
    """
    Applies crop to a Pillow image.
    crop_rect is [x, y, w, h]. Pillow's crop box is (left, upper, right, lower).
    """
    if not (isinstance(crop_rect, list) and len(crop_rect) == 4 and all(isinstance(n, int) for n in crop_rect)):
        raise ValueError("crop_rect must be a list of 4 integers [x, y, w, h]")

    x, y, w, h = crop_rect
    left = x
    upper = y
    right = x + w
    lower = y + h

    # Ensure crop box is within image bounds
    img_width, img_height = pil_img.size
    left = max(0, left)
    upper = max(0, upper)
    right = min(img_width, right)
    lower = min(img_height, lower)

    if left >= right or upper >= lower:
        # This means the crop rectangle is invalid or outside the image.
        # Depending on desired behavior, could return original, raise error, or return empty.
        # For now, let's raise an error if the crop rect becomes non-positive area after boundary checks.
        raise ImageProcessingError(f"Invalid crop rectangle [{left},{upper},{right},{lower}] after boundary adjustment for image size {pil_img.size}.")


    try:
        return pil_img.crop((left, upper, right, lower))
    except Exception as e:
        raise ImageProcessingError(f"Error applying crop: {e}")

def apply_face_smoothing(
    pil_img: Image.Image,
    face_detection_results: List[Dict[str, Any]],
    intensity: float # intensity from 0.0 to 1.0
) -> Image.Image:
    """
    Applies smoothing to detected face regions using bilateral filter.
    Intensity controls the filter parameters (d, sigmaColor, sigmaSpace).
    """
    if not face_detection_results or intensity == 0.0:
        return pil_img

    img_cv = pil_to_cv2(pil_img)

    # Parameters for bilateralFilter:
    # d: Diameter of each pixel neighborhood.
    #    Larger d means more pixels are included in the average.
    # sigmaColor: Filter sigma in the color space.
    #    A larger value means farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
    # sigmaSpace: Filter sigma in the coordinate space.
    #    A larger value means that pixels farther out from the central pixel will influence the filtering, as long as their colors are close enough (see sigmaColor).
    #
    # We can scale these based on 'intensity'.
    # These are example scalings, might need tuning.
    d = int(9 * intensity) # Max d around 9-15 is common for noticeable smoothing
    if d < 1: d = 1 # d must be positive for cv2.bilateralFilter
    sigma_color = int(75 * intensity)
    sigma_space = int(75 * intensity)
    if sigma_color < 1: sigma_color = 1
    if sigma_space < 1: sigma_space = 1


    for face in face_detection_results:
        box = face.get('box') # [x, y, w, h]
        if not box or len(box) != 4:
            continue

        x, y, w, h = box
        fx, fy, fw, fh = int(x), int(y), int(w), int(h) # Ensure integer coordinates

        # Ensure the ROI is within image bounds
        img_h, img_w = img_cv.shape[:2]
        roi_x1, roi_y1 = max(0, fx), max(0, fy)
        roi_x2, roi_y2 = min(img_w, fx + fw), min(img_h, fy + fh)

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2: # Skip if ROI is invalid
            continue

        face_roi = img_cv[roi_y1:roi_y2, roi_x1:roi_x2]

        if face_roi.size == 0: # Skip if ROI is empty
            continue

        # Apply bilateral filter to the face ROI
        # Important: bilateralFilter can be slow on large images or large d.
        # For performance, consider applying to a slightly upscaled version of ROI then downscale,
        # or ensure 'd' is not excessively large.
        smoothed_face_roi = cv2.bilateralFilter(face_roi, d, sigma_color, sigma_space)

        # Place the smoothed ROI back into the main image
        img_cv[roi_y1:roi_y2, roi_x1:roi_x2] = smoothed_face_roi

    return cv2_to_pil(img_cv)

def apply_selective_background_blur(
    pil_img: Image.Image,
    blur_radius: int # e.g., 3, 5, etc. This will be kernel size for GaussianBlur
) -> Image.Image:
    """
    Applies Gaussian blur to the background of an image, keeping the subject in focus.
    Uses MediaPipe Selfie Segmentation.
    """
    if blur_radius == 0:
        return pil_img

    img_cv = pil_to_cv2(pil_img)
    # MediaPipe works with RGB images. Ensure img_cv is in BGR for OpenCV, then convert for MediaPipe.
    # If img_cv has 4 channels (BGRA from RGBA PIL image), handle it:
    if img_cv.shape[2] == 4:
        img_bgr_for_mediapipe = img_cv[:,:,:3] # Take BGR part
    else:
        img_bgr_for_mediapipe = img_cv

    img_rgb_for_mediapipe = cv2.cvtColor(img_bgr_for_mediapipe, cv2.COLOR_BGR2RGB)


    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    # Using a try-finally block to ensure resources are released
    # according to MediaPipe documentation best practices.
    output_image_cv = img_cv # Default to original if segmentation fails early
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        # Process the image and get the segmentation mask
        results = selfie_segmentation.process(img_rgb_for_mediapipe)

        # The mask is a float32 NumPy array where values are typically 0.0 (background) to 1.0 (person).
        # results.segmentation_mask shape will be (height, width)
        mask = results.segmentation_mask

        if mask is None:
            # Fallback or error if segmentation fails
            # TODO: Replace print with actual logger
            print("Warning: Selfie segmentation failed to produce a mask.")
            return pil_img # Return original image

        # Create a blurred version of the original image
        # GaussianBlur kernel size must be odd. Adjust blur_radius if it's even.
        kernel_size = blur_radius * 2 + 1 # Ensure it's odd, e.g. radius 3 -> kernel 7x7

        # Apply Gaussian blur to the original image (img_cv, which is BGR or BGRA)
        blurred_image_cv = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)

        # Create a multi-channel mask for combining, matching the number of channels in img_cv
        # The mask from MediaPipe is single channel. We need to stack it for color/alpha images.
        num_channels = img_cv.shape[2]
        condition = np.stack((mask,) * num_channels, axis=-1) > BLUR_MASK_THRESHOLD

        # Combine the original foreground with the blurred background
        # Where condition is True, take from original_image, else from blurred_image
        output_image_cv = np.where(condition, img_cv, blurred_image_cv)

    return cv2_to_pil(output_image_cv)

# Main orchestration function - to be expanded
def apply_enhancements(
    image_path: str,
    params: Dict[str, Any], # Corresponds to EnhancementRequestParams
    face_detection_results: Optional[List[Dict[str, Any]]] = None # Optional for now
) -> Image.Image: # Returns a Pillow Image object
    """
    Applies a set of enhancement parameters to an image.
    Order of operations can be important. A common order:
    1. Load image
    2. Adjustments (Brightness, Contrast, Saturation)
    3. Face specific adjustments (Smoothing) - requires face data
    4. Selective blur (Background blur) - requires segmentation
    5. Crop (often last, or sometimes first to reduce processing area) - let's do it towards the end.
    """
    pil_img = load_image_pil(image_path)

    # 1. Apply Brightness, Contrast, Saturation
    pil_img = apply_brightness_contrast_saturation(
        pil_img,
        params.get('brightness_target', 1.0),
        params.get('contrast_target', 1.0),
        params.get('saturation_target', 1.0)
    )

    # --- Face Smoothing ---
    face_smooth_intensity = params.get('face_smooth_intensity', 0.0)
    # Ensure face_detection_results is not None before passing
    current_face_detection_results = face_detection_results if face_detection_results is not None else []

    if face_smooth_intensity > 0 and current_face_detection_results:
        try:
            pil_img = apply_face_smoothing(pil_img, current_face_detection_results, face_smooth_intensity)
        except Exception as e:
            # Log this error, but maybe don't fail the whole process? Or re-raise as ImageProcessingError
            # For now, let's assume if smoothing fails, we can proceed without it.
            # TODO: Replace print with actual logger
            print(f"Warning: Face smoothing failed: {e}")
            # raise ImageProcessingError(f"Face smoothing failed: {e}") # Or re-raise

    # --- Background Blur ---
    background_blur_radius = params.get('background_blur_radius', 0)
    if background_blur_radius > 0:
        try:
            pil_img = apply_selective_background_blur(pil_img, background_blur_radius)
        except Exception as e:
            # TODO: Replace print with actual logger
            print(f"Warning: Background blur failed: {e}")
            # raise ImageProcessingError(f"Background blur failed: {e}") # Or re-raise

    # 2. Apply Crop
    crop_rect = params.get('crop_rect')
    if crop_rect: # crop_rect should always be there based on B008, but good to check
        # Ensure crop_rect is valid for the current image dimensions (after potential other ops, though unlikely here)
        # For simplicity, we assume crop_rect was calculated on original dimensions and is still mostly valid.
        # More robust would be to re-validate crop_rect against current pil_img.size if other ops changed dimensions.
        pil_img = apply_crop(pil_img, crop_rect)

    return pil_img

if __name__ == '__main__':
    # Example usage (for testing purposes - requires creating dummy image and params)
    print("Image processing service module. Run tests for example usage.")
    # try:
    #     # Create a dummy image for testing
    #     dummy_img_path = "dummy_test_image.png"
    #     img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    #     img_array[:,:] = [255,128,0] # Orange
    #     cv2.imwrite(dummy_img_path, img_array)

    #     test_params = {
    #         "brightness_target": 1.2,
    #         "contrast_target": 1.1,
    #         "saturation_target": 1.3,
    #         "crop_rect": [10, 10, 50, 50], # x, y, w, h
    #         "face_smooth_intensity": 0.0, # Not testing yet
    #         "background_blur_radius": 0 # Not testing yet
    #     }
    #     processed_pil_img = apply_enhancements(dummy_img_path, test_params)
    #     processed_pil_img.save("dummy_processed_image.png")
    #     print(f"Processed image saved to dummy_processed_image.png")

    #     # Clean up
    #     os.remove(dummy_img_path)
    #     # os.remove("dummy_processed_image.png") # Keep it to inspect
    # except Exception as e:
    #     print(f"Error in example usage: {e}")
