import pytest
import numpy as np
import cv2
import os
import tempfile

from backend.services.apply_image_modifications import (
    apply_brightness_contrast,
    apply_saturation,
    apply_crop,
    apply_background_blur,
    apply_face_smoothing,
    apply_all_enhancements
)

# --- Helper Functions ---

def create_dummy_image(height: int, width: int, channels: int = 3, color: tuple = (128, 128, 128), dtype=np.uint8) -> np.ndarray:
    """Creates a dummy image with a solid color."""
    if channels == 1:
        img = np.full((height, width), color[0], dtype=dtype)
    else:
        img = np.full((height, width, channels), color, dtype=dtype)
    return img

def create_gradient_image(height: int, width: int) -> np.ndarray:
    """Creates a dummy gradient image for saturation tests."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = int(j * 255 / width)  # Blue channel gradient
            img[i, j, 1] = int(i * 255 / height) # Green channel gradient
            img[i, j, 2] = 128 # Red channel constant
    return img

# --- Tests for apply_brightness_contrast ---

def test_bc_no_change():
    img = create_dummy_image(10, 10)
    modified_img = apply_brightness_contrast(img.copy(), brightness=0.0, contrast=1.0)
    assert np.array_equal(img, modified_img)

def test_bc_increase_brightness():
    img = create_dummy_image(10, 10, color=(100, 100, 100))
    modified_img = apply_brightness_contrast(img.copy(), brightness=50.0, contrast=1.0)
    assert np.all(modified_img >= img)
    assert np.mean(modified_img) > np.mean(img)
    assert np.all(modified_img <= 255)

def test_bc_decrease_brightness():
    img = create_dummy_image(10, 10, color=(100, 100, 100))
    modified_img = apply_brightness_contrast(img.copy(), brightness=-50.0, contrast=1.0)
    assert np.all(modified_img <= img)
    assert np.mean(modified_img) < np.mean(img)
    assert np.all(modified_img >= 0)

def test_bc_increase_contrast():
    img = create_dummy_image(10, 10, color=(128,128,128))
    # Add some variation for contrast to be meaningful
    img[0:5,0:5] = (100,100,100)
    img[5:10,5:10] = (150,150,150)
    original_std = np.std(img)
    modified_img = apply_brightness_contrast(img.copy(), brightness=0.0, contrast=1.5)
    assert np.std(modified_img) > original_std

def test_bc_decrease_contrast():
    img = create_dummy_image(10, 10, color=(128,128,128))
    img[0:5,0:5] = (50,50,50)
    img[5:10,5:10] = (200,200,200)
    original_std = np.std(img)
    modified_img = apply_brightness_contrast(img.copy(), brightness=0.0, contrast=0.5)
    assert np.std(modified_img) < original_std

def test_bc_value_error_on_none_image():
    with pytest.raises(ValueError, match="Input image cannot be None"):
        apply_brightness_contrast(None, 0.0, 1.0)


# --- Tests for apply_saturation ---

def test_saturation_no_change():
    img = create_gradient_image(10, 10)
    modified_img = apply_saturation(img.copy(), saturation_factor=1.0)
    assert np.array_equal(img, modified_img)

def test_saturation_increase():
    img = create_gradient_image(10, 10) # Use gradient for more obvious saturation change
    modified_img = apply_saturation(img.copy(), saturation_factor=1.5)
    # Increased saturation typically changes pixel values
    assert not np.array_equal(img, modified_img)
    # Check if S channel values are generally higher (or clipped)
    hsv_original = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_modified = cv2.cvtColor(modified_img, cv2.COLOR_BGR2HSV)
    assert np.mean(hsv_modified[:,:,1]) >= np.mean(hsv_original[:,:,1]) # or a more nuanced check

def test_saturation_decrease_to_grayscale():
    img = create_gradient_image(10, 10)
    modified_img = apply_saturation(img.copy(), saturation_factor=0.0)
    # In grayscale, R, G, and B values should be very close to each other for each pixel
    # (or perfectly equal if conversion is perfect)
    # A common check is that the S channel in HSV is close to 0
    hsv_modified = cv2.cvtColor(modified_img, cv2.COLOR_BGR2HSV)
    assert np.all(hsv_modified[:,:,1] < 5) # S channel should be near 0 for grayscale

def test_saturation_value_error_on_none_image():
    with pytest.raises(ValueError, match="Input image cannot be None"):
        apply_saturation(None, 1.0)

# --- Tests for apply_crop ---

def test_crop_valid():
    img = create_dummy_image(20, 20)
    crop_rect = [5, 5, 10, 10] # x, y, w, h
    modified_img = apply_crop(img.copy(), crop_rect)
    assert modified_img.shape == (10, 10, 3)

def test_crop_full_image():
    img = create_dummy_image(20, 20)
    crop_rect = [0, 0, 20, 20]
    modified_img = apply_crop(img.copy(), crop_rect)
    assert np.array_equal(img, modified_img)

def test_crop_outside_bounds():
    img = create_dummy_image(20, 20)
    crop_rect = [15, 15, 10, 10] # Partially outside
    modified_img = apply_crop(img.copy(), crop_rect)
    assert modified_img.shape == (5, 5, 3) # Clipped to (20-15, 20-15)

def test_crop_zero_wh():
    img = create_dummy_image(20, 20)
    crop_rect_zero_w = [5, 5, 0, 10]
    crop_rect_zero_h = [5, 5, 10, 0]
    # Current implementation returns original image if adjusted w/h is <=0
    assert np.array_equal(apply_crop(img.copy(), crop_rect_zero_w), img)
    assert np.array_equal(apply_crop(img.copy(), crop_rect_zero_h), img)


def test_crop_invalid_rect_value_error():
    img = create_dummy_image(10,10)
    with pytest.raises(ValueError, match="Invalid crop_rect"):
        apply_crop(img, [])
    with pytest.raises(ValueError, match="Crop width and height must be positive"):
        apply_crop(img, [0,0,-5,5])


def test_crop_value_error_on_none_image():
    with pytest.raises(ValueError, match="Input image cannot be None"):
        apply_crop(None, [0,0,1,1])

# --- Tests for apply_background_blur (full image blur) ---

def test_blur_radius_zero():
    img = create_dummy_image(10, 10)
    modified_img = apply_background_blur(img.copy(), blur_radius_pixels=0)
    assert np.array_equal(img, modified_img)

def test_blur_radius_one(): # Also treated as no-op by current function
    img = create_dummy_image(10, 10)
    modified_img = apply_background_blur(img.copy(), blur_radius_pixels=1)
    assert np.array_equal(img, modified_img)

def test_blur_applied():
    img = create_dummy_image(10, 10)
    modified_img = apply_background_blur(img.copy(), blur_radius_pixels=3) # Kernel (7,7)
    assert not np.array_equal(img, modified_img)
    # A blurred image will likely have different mean if edges are blurred towards a color
    # More robust: check if high-frequency components are reduced (e.g. std dev of laplacian)
    # For a solid color image, blur has no visible effect but values might change due to float precision
    # Let's use a gradient image for blur test
    gradient_img = create_gradient_image(30,30)
    blurred_gradient_img = apply_background_blur(gradient_img.copy(), blur_radius_pixels=3)
    assert not np.array_equal(gradient_img, blurred_gradient_img)
    # Check if edges are less sharp, e.g. difference between adjacent pixels is smaller on average
    # This is a simplistic check for blur
    original_diff = np.mean(np.abs(np.diff(gradient_img.astype(np.int16), axis=0)))
    blurred_diff = np.mean(np.abs(np.diff(blurred_gradient_img.astype(np.int16), axis=0)))
    assert blurred_diff < original_diff


def test_blur_value_error_on_none_image():
    with pytest.raises(ValueError, match="Input image cannot be None"):
        apply_background_blur(None, 5)

# --- Tests for apply_face_smoothing ---

def test_smoothing_intensity_zero():
    img = create_dummy_image(20, 20)
    face_boxes = [[5, 5, 10, 10]]
    modified_img = apply_face_smoothing(img.copy(), intensity=0.0, face_boxes=face_boxes)
    assert np.array_equal(img, modified_img)

def test_smoothing_no_face_boxes():
    img = create_dummy_image(20, 20)
    modified_img_none = apply_face_smoothing(img.copy(), intensity=0.5, face_boxes=None)
    modified_img_empty = apply_face_smoothing(img.copy(), intensity=0.5, face_boxes=[])
    assert np.array_equal(img, modified_img_none)
    assert np.array_equal(img, modified_img_empty)

def test_smoothing_applied_to_face_only():
    img = create_dummy_image(30, 30, color=(128, 128, 128))
    # Make face region different to see effect
    img[5:15, 5:15] = (100, 150, 200)
    face_box = [5, 5, 10, 10] # x, y, w, h

    original_face_roi = img[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]].copy()
    original_non_face_roi_corner = img[0:5, 0:5].copy()

    modified_img = apply_face_smoothing(img.copy(), intensity=0.8, face_boxes=[face_box])

    modified_face_roi = modified_img[face_box[1]:face_box[1]+face_box[3], face_box[0]:face_box[0]+face_box[2]]
    modified_non_face_roi_corner = modified_img[0:5, 0:5]

    assert not np.array_equal(original_face_roi, modified_face_roi), "Face ROI should be changed by smoothing"
    assert np.array_equal(original_non_face_roi_corner, modified_non_face_roi_corner), "Non-face ROI should NOT be changed"

def test_smoothing_value_error_on_none_image():
    with pytest.raises(ValueError, match="Input image cannot be None"):
        apply_face_smoothing(None, 0.5, [[0,0,1,1]])

# --- Tests for apply_all_enhancements ---

@pytest.fixture
def temp_image_file():
    img = create_dummy_image(50, 50, color=(100, 150, 200))
    # Use NamedTemporaryFile to handle cleanup
    # temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    # Using mkstemp for more control over name if needed, but NamedTemporaryFile is usually fine
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd) # Close the file descriptor so cv2.imwrite can use the path
    cv2.imwrite(path, img)
    yield path
    os.remove(path)

def test_all_enhancements_loads_and_processes(temp_image_file):
    params = {
        "brightness_target": 20.0,
        "contrast_target": 1.2,
        "saturation_target": 1.3,
        "background_blur_radius": 2,
        "crop_rect": [5, 5, 40, 40],
        "face_smooth_intensity": 0.6,
        "face_boxes": [[10, 10, 20, 20]] # Relative to cropped image if crop is first
    }
    # Note: apply_all_enhancements applies crop first. Face boxes should be relative to original image
    # or the service needs to adjust them. The current apply_all_enhancements doesn't adjust face_boxes post-crop.
    # For this test, let's assume face_boxes are relative to original image and test post-crop.
    # The current implementation of `apply_all_enhancements` uses `face_boxes` as is.
    # If crop is applied first, then `face_boxes` coords need to be relative to the cropped region or adjusted.
    # For simplicity, let's make face_boxes relative to the crop for this test.
    # Original image 50x50. Crop to 40x40 from (5,5).
    # Face box [10,10,20,20] in original image becomes [5,5,20,20] in cropped image.
    # However, apply_all_enhancements takes original face_boxes.
    # Let's adjust face_boxes in params to be relative to the original image, suitable for post-crop processing.
    # Crop: [5,5,40,40]. Face box [10,10,20,20] (original coords)
    # After crop, image is 40x40. Face box relative to this new image: [10-5, 10-5, 20, 20] = [5,5,20,20]
    # The `apply_all_enhancements` function doesn't currently transform face_box coordinates after crop.
    # This means `apply_face_smoothing` will use original coordinates on the (potentially) cropped image.
    # This is a known limitation to address if this were production. For unit test, test current behavior.

    modified_image = apply_all_enhancements(temp_image_file, params)
    assert modified_image is not None
    assert modified_image.shape == (40, 40, 3) # Due to crop [x,y,w,h] -> (h,w,c)

    original_image = cv2.imread(temp_image_file)
    assert not np.array_equal(original_image[5:45, 5:45], modified_image) # Check if it changed from cropped original

def test_all_enhancements_non_existent_file():
    params = {}
    modified_image = apply_all_enhancements("non_existent_image.png", params)
    assert modified_image is None

def test_all_enhancements_no_params(temp_image_file):
    params = {} # Empty params, should use defaults (mostly no-ops)
    modified_image = apply_all_enhancements(temp_image_file, params)
    assert modified_image is not None
    original_image = cv2.imread(temp_image_file)
    # With default params (brightness=0, contrast=1, sat=1, blur=0, smooth=0, no crop), should be same
    assert np.array_equal(original_image, modified_image)

def test_all_enhancements_only_blur(temp_image_file):
    params = {"background_blur_radius": 3}
    modified_image = apply_all_enhancements(temp_image_file, params)
    assert modified_image is not None
    original_image = cv2.imread(temp_image_file)
    assert not np.array_equal(original_image, modified_image) # Should be different due to blur


if __name__ == '__main__':
    pass
    # For local testing:
    # Create a dummy_image.png (e.g. 100x100, BGR) in the same directory as this test file.
    # img_for_test = create_dummy_image(100,100, color=(50,100,150))
    # cv2.imwrite("dummy_image.png", img_for_test)
    # print("Created dummy_image.png for testing apply_all_enhancements")
    #
    # params_example = {
    #    "brightness_target": 30.0,
    #    "contrast_target": 1.3,
    #    "saturation_target": 1.5,
    #    "background_blur_radius": 3,
    #    "crop_rect": [10, 10, 80, 80],
    #    "face_smooth_intensity": 0.7,
    #    "face_boxes": [[20,20,40,40]] # These are coords on original image
    # }
    # try:
    #    modified_example = apply_all_enhancements("dummy_image.png", params_example)
    #    if modified_example is not None:
    #        cv2.imwrite("modified_dummy_image.png", modified_example)
    #        print("SUCCESS: Processed dummy_image.png -> modified_dummy_image.png")
    #    else:
    #        print("FAILURE: apply_all_enhancements returned None for dummy_image.png")
    # except Exception as e:
    #    print(f"FAILURE: Error during local test of apply_all_enhancements: {e}")
