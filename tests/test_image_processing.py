import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import numpy as np
from PIL import Image, ImageEnhance

# Add the services directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock MediaPipe before it's imported by image_processing module
# This is crucial if MediaPipe is imported at the module level in image_processing.py
mock_mp = MagicMock()
sys.modules['mediapipe'] = mock_mp
sys.modules['mediapipe.solutions'] = mock_mp.solutions
sys.modules['mediapipe.solutions.selfie_segmentation'] = mock_mp.solutions.selfie_segmentation

# Import the module to be tested
from services import image_processing


class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        # Create a dummy PIL image for testing
        self.dummy_pil_image = Image.new('RGB', (100, 100), color = 'red')
        self.dummy_image_path = "dummy_test_image.png"
        self.dummy_pil_image.save(self.dummy_image_path)

        # Mock face detection results
        self.mock_faces = [{'box': [10, 10, 30, 30], 'confidence': 1.0}] # x, y, w, h

        # Mock parameters for enhancement
        self.base_params = {
            "brightness_target": 1.0,
            "contrast_target": 1.0,
            "saturation_target": 1.0,
            "background_blur_radius": 0,
            "crop_rect": [0, 0, 100, 100], # Full image
            "face_smooth_intensity": 0.0,
        }

    def tearDown(self):
        if os.path.exists(self.dummy_image_path):
            os.remove(self.dummy_image_path)
        if os.path.exists("processed_test_image.png"): # Just in case any test saves this
            os.remove("processed_test_image.png")

    @patch('services.image_processing.Image.open')
    def test_load_image_pil_success(self, mock_image_open):
        mock_img = MagicMock(spec=Image.Image)
        mock_img.mode = 'RGB'
        mock_img.info = {}
        mock_image_open.return_value = mock_img

        with patch('services.image_processing.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            img = image_processing.load_image_pil("fake_path.jpg")
            self.assertIsNotNone(img)
            mock_image_open.assert_called_with("fake_path.jpg")

    @patch('services.image_processing.os.path.exists')
    def test_load_image_pil_not_found(self, mock_exists):
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            image_processing.load_image_pil("non_existent.jpg")

    @patch('services.image_processing.cv2.imread')
    def test_load_image_cv2_success(self, mock_cv2_imread):
        mock_img_cv = np.zeros((100,100,3), dtype=np.uint8)
        mock_cv2_imread.return_value = mock_img_cv
        with patch('services.image_processing.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            img = image_processing.load_image_cv2("fake_path.jpg")
            self.assertIsNotNone(img)
            mock_cv2_imread.assert_called_with("fake_path.jpg", image_processing.cv2.IMREAD_UNCHANGED)

    def test_pil_to_cv2_rgb(self):
        pil_img = Image.new('RGB', (10, 20), color='blue')
        cv2_img = image_processing.pil_to_cv2(pil_img)
        self.assertEqual(cv2_img.shape, (20, 10, 3)) # h, w, c
        self.assertTrue(np.array_equal(cv2_img[0,0], [0,0,255])) # Check for BGR blue

    def test_cv2_to_pil_bgr(self):
        cv2_img = np.zeros((20,10,3), dtype=np.uint8)
        cv2_img[:,:,0] = 255 # Blue channel in BGR
        pil_img = image_processing.cv2_to_pil(cv2_img)
        self.assertEqual(pil_img.mode, 'RGB')
        self.assertEqual(pil_img.size, (10,20)) # w, h
        self.assertEqual(pil_img.getpixel((0,0)), (0,0,255)) # Check for RGB blue


    def test_apply_brightness_contrast_saturation(self):
        img = self.dummy_pil_image.copy()
        # Test with factors that should cause change
        processed_img = image_processing.apply_brightness_contrast_saturation(img, 1.5, 1.2, 1.3)
        self.assertNotEqual(list(img.getdata()), list(processed_img.getdata()))


        # Test with factors that should not cause change (1.0)
        processed_img_no_change = image_processing.apply_brightness_contrast_saturation(img, 1.0, 1.0, 1.0)
        self.assertEqual(list(img.getdata()), list(processed_img_no_change.getdata()))


    def test_apply_crop(self):
        img = self.dummy_pil_image.copy() # 100x100
        crop_rect = [10, 10, 50, 50] # x, y, w, h
        cropped_img = image_processing.apply_crop(img, crop_rect)
        self.assertEqual(cropped_img.size, (50, 50)) # w, h

    def test_apply_crop_invalid_rect_too_large(self):
        img = self.dummy_pil_image.copy() #100x100
        # This crop will be adjusted to fit, not raise error, as per current apply_crop logic
        # crop_rect [x,y,w,h] -> Pillow crop (left, upper, right, lower)
        # [0,0,200,200] -> (0,0,200,200), adjusted to (0,0,100,100) for a 100x100 image
        cropped_img = image_processing.apply_crop(img, [0,0,200,200])
        self.assertEqual(cropped_img.size, (100,100))

    def test_apply_crop_invalid_rect_non_positive_area(self):
        img = self.dummy_pil_image.copy() #100x100
        # Example: x=10, y=10, w=0, h=50. This becomes left=10, upper=10, right=10, lower=60. left >= right.
        with self.assertRaisesRegex(image_processing.ImageProcessingError, "Invalid crop rectangle"):
            image_processing.apply_crop(img, [10,10,0,50])

        # Example: right or lower ends up less than left or upper after boundary adjustment
        # Image is 100x100. Crop from x=110 (way outside)
        # crop_rect = [110, 10, 50, 50] -> left=110, upper=10, right=160, lower=60
        # Adjusted: left=max(0,110)=110. right=min(100,160)=100. Now left (110) > right (100)
        with self.assertRaisesRegex(image_processing.ImageProcessingError, "Invalid crop rectangle"):
            image_processing.apply_crop(img, [110,10,50,50])


    # Test apply_enhancements - basic run without complex ops
    @patch('services.image_processing.load_image_pil')
    @patch('services.image_processing.apply_brightness_contrast_saturation')
    @patch('services.image_processing.apply_crop')
    @patch('services.image_processing.apply_face_smoothing') # Mock these for now
    @patch('services.image_processing.apply_selective_background_blur') # Mock these
    def test_apply_enhancements_orchestration_basic(
        self, mock_blur, mock_smooth, mock_crop, mock_bcs, mock_load_pil
    ):
        mock_load_pil.return_value = self.dummy_pil_image.copy()
        mock_bcs.return_value = self.dummy_pil_image.copy()
        mock_crop.return_value = self.dummy_pil_image.copy()
        mock_smooth.return_value = self.dummy_pil_image.copy() # Ensure it returns a PIL image
        mock_blur.return_value = self.dummy_pil_image.copy()  # Ensure it returns a PIL image


        params = self.base_params.copy()
        params["brightness_target"] = 1.1 # to trigger BCS call path

        result_img = image_processing.apply_enhancements(self.dummy_image_path, params, self.mock_faces)

        mock_load_pil.assert_called_with(self.dummy_image_path)
        mock_bcs.assert_called_once()
        # Smoothing and blur are not called if intensity/radius is 0
        mock_smooth.assert_not_called() # Since face_smooth_intensity is 0.0 in base_params
        mock_blur.assert_not_called()   # Since background_blur_radius is 0 in base_params
        mock_crop.assert_called_once()
        self.assertIsNotNone(result_img)

    # In TestImageProcessing class:

    @patch('services.image_processing.cv2.bilateralFilter')
    def test_apply_face_smoothing_with_faces(self, mock_bilateral_filter):
        # Make the mock filter return the input image region to simplify checking calls
        mock_bilateral_filter.side_effect = lambda roi, d, sc, ss: roi

        img = self.dummy_pil_image.copy()
        # Ensure intensity is high enough to trigger the filter
        processed_img = image_processing.apply_face_smoothing(img, self.mock_faces, 0.5)

        mock_bilateral_filter.assert_called() # Check it was called at least once
        # We could add more assertions here, e.g., number of calls based on mock_faces
        self.assertIsNotNone(processed_img)
        # As bilateralFilter is mocked to return the original ROI, the image content shouldn't change.
        # This primarily tests the logic for ROI extraction and filter application path.
        self.assertEqual(list(img.getdata()), list(processed_img.getdata()))


    def test_apply_face_smoothing_no_faces(self):
        img = self.dummy_pil_image.copy()
        processed_img = image_processing.apply_face_smoothing(img, [], 0.5)
        # Should return original image if no faces
        self.assertEqual(list(img.getdata()), list(processed_img.getdata()))

    def test_apply_face_smoothing_zero_intensity(self):
        img = self.dummy_pil_image.copy()
        processed_img = image_processing.apply_face_smoothing(img, self.mock_faces, 0.0)
        # Should return original image if intensity is zero
        self.assertEqual(list(img.getdata()), list(processed_img.getdata()))

    @patch('services.image_processing.mp.solutions.selfie_segmentation') # Mock the top-level import
    @patch('services.image_processing.cv2.GaussianBlur')
    def test_apply_selective_background_blur_with_radius(self, mock_gaussian_blur, mock_selfie_segmentation_module):
        # Configure the mock for MediaPipe SelfieSegmentation
        mock_segmentation_instance = MagicMock()
        mock_selfie_segmentation_module.SelfieSegmentation.return_value.__enter__.return_value = mock_segmentation_instance

        # Create a dummy mask (e.g., person in the center)
        # Mask should be HxW, float32, values 0.0 to 1.0
        dummy_mask = np.zeros((100, 100), dtype=np.float32)
        dummy_mask[25:75, 25:75] = 1.0 # Person area
        mock_segmentation_instance.process.return_value.segmentation_mask = dummy_mask

        # Mock GaussianBlur to return a distinctly different image (e.g., all black)
        # to verify it was used in np.where
        blurred_cv_image_content = np.full((100, 100, 3), 0, dtype=np.uint8) # Black image
        mock_gaussian_blur.return_value = blurred_cv_image_content

        img_pil = self.dummy_pil_image.copy() # 100x100, red
        blur_radius = 5
        processed_pil_img = image_processing.apply_selective_background_blur(img_pil, blur_radius)

        mock_selfie_segmentation_module.SelfieSegmentation.assert_called_once()
        mock_segmentation_instance.process.assert_called_once()

        # GaussianBlur kernel size should be blur_radius * 2 + 1
        expected_kernel_size = (blur_radius * 2 + 1, blur_radius * 2 + 1)
        mock_gaussian_blur.assert_called_with(unittest.mock.ANY, expected_kernel_size, 0)

        # Verify that the output image is a combination:
        # Original red where mask=1 (person), blurred content (black) where mask=0 (background)
        processed_cv_img = image_processing.pil_to_cv2(processed_pil_img)

        # Check a pixel from the "person" area (should be original red: [0,0,255] in BGR)
        self.assertTrue(np.array_equal(processed_cv_img[50,50], [0,0,255]))
        # Check a pixel from the "background" area (should be from blurred_cv_image_content: [0,0,0] in BGR)
        self.assertTrue(np.array_equal(processed_cv_img[10,10], [0,0,0]))


    def test_apply_selective_background_blur_zero_radius(self):
        img = self.dummy_pil_image.copy()
        processed_img = image_processing.apply_selective_background_blur(img, 0)
        # Should return original image if blur_radius is zero
        self.assertEqual(list(img.getdata()), list(processed_img.getdata()))

    @patch('services.image_processing.mp.solutions.selfie_segmentation')
    def test_apply_selective_background_blur_segmentation_fails(self, mock_selfie_segmentation_module):
        # Simulate segmentation failure (mask is None)
        mock_segmentation_instance = MagicMock()
        mock_selfie_segmentation_module.SelfieSegmentation.return_value.__enter__.return_value = mock_segmentation_instance
        mock_segmentation_instance.process.return_value.segmentation_mask = None

        img = self.dummy_pil_image.copy()
        with patch('builtins.print') as mock_print: # Suppress/check warning
            processed_img = image_processing.apply_selective_background_blur(img, 5)
            mock_print.assert_any_call("Warning: Selfie segmentation failed to produce a mask.")

        # Should return original image if segmentation fails
        self.assertEqual(list(img.getdata()), list(processed_img.getdata()))


    # Update the comprehensive apply_enhancements test
    @patch('services.image_processing.load_image_pil')
    @patch('services.image_processing.apply_brightness_contrast_saturation')
    @patch('services.image_processing.apply_crop')
    @patch('services.image_processing.apply_face_smoothing')
    @patch('services.image_processing.apply_selective_background_blur')
    def test_apply_enhancements_orchestration_all_ops(
        self, mock_blur, mock_smooth, mock_crop, mock_bcs, mock_load_pil
    ):
        # Setup mocks to return distinct PIL images to trace calls if needed, or just copies
        mock_load_pil.return_value = self.dummy_pil_image.copy()
        mock_bcs.return_value = Image.new("RGB", (100,100), "blue") # To track change
        mock_smooth.return_value = Image.new("RGB", (100,100), "green")
        mock_blur.return_value = Image.new("RGB", (100,100), "yellow")
        mock_crop.return_value = Image.new("RGB", (50,50), "black") # Crop changes size

        params = self.base_params.copy()
        params["brightness_target"] = 1.1
        params["face_smooth_intensity"] = 0.5
        params["background_blur_radius"] = 3
        params["crop_rect"] = [0,0,50,50] # Different from base to ensure it's used

        # Provide face detection results to trigger smoothing
        face_results = self.mock_faces

        result_img = image_processing.apply_enhancements(self.dummy_image_path, params, face_results)

        mock_load_pil.assert_called_with(self.dummy_image_path)
        mock_bcs.assert_called_once()
        mock_smooth.assert_called_with(unittest.mock.ANY, face_results, params["face_smooth_intensity"])
        mock_blur.assert_called_with(unittest.mock.ANY, params["background_blur_radius"])
        mock_crop.assert_called_with(unittest.mock.ANY, params["crop_rect"])

        self.assertIsNotNone(result_img)
        self.assertEqual(result_img.size, (50,50)) # Check final image is the one from crop
        self.assertEqual(result_img.getpixel((0,0)), (0,0,0)) # Black, from mocked crop


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
