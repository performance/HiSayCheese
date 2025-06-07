import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the services directory to the Python path for import
# This assumes the test is run from the root directory of the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.auto_enhancement import calculate_auto_enhancements

class TestAutoEnhancement(unittest.TestCase):

    def setUp(self):
        self.image_path = "dummy/path/to/image.jpg"
        self.image_dimensions = (1000, 800) # width, height
        self.mock_face_results_none = []
        self.mock_face_results_one_face = [
            {'box': [100, 100, 200, 250], 'confidence': 1.0} # x, y, w, h
        ]
        self.mock_face_results_two_faces = [
            {'box': [100, 100, 200, 250], 'confidence': 1.0}, # area = 50000
            {'box': [500, 300, 300, 350], 'confidence': 1.0}  # area = 105000 (larger)
        ]

    def test_brightness_dark_image(self):
        quality_results = {'brightness': 0.2, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['brightness_target'], 1.5)

    def test_brightness_bright_image(self):
        quality_results = {'brightness': 0.8, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['brightness_target'], 0.8)

    def test_brightness_normal_image(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['brightness_target'], 1.1)

    def test_brightness_passport_dark(self):
        quality_results = {'brightness': 0.2, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results, mode="passport")
        self.assertEqual(params['brightness_target'], 1.2) # Less aggressive for passport

    def test_contrast_low_contrast(self):
        quality_results = {'brightness': 0.5, 'contrast': 10} # Low contrast
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['contrast_target'], 1.2)

    def test_contrast_normal_contrast(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['contrast_target'], 1.05)

    def test_contrast_passport_low(self):
        quality_results = {'brightness': 0.5, 'contrast': 10}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results, mode="passport")
        self.assertEqual(params['contrast_target'], 1.1) # Less aggressive

    def test_saturation_default(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['saturation_target'], 1.1)

    def test_saturation_passport(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results, mode="passport")
        self.assertEqual(params['saturation_target'], 1.0)

    def test_background_blur_no_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['background_blur_radius'], 0)

    def test_background_blur_with_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results)
        self.assertEqual(params['background_blur_radius'], 3)

    def test_background_blur_passport_with_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results, mode="passport")
        self.assertEqual(params['background_blur_radius'], 0) # No blur for passport

    def test_face_smoothing_no_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        self.assertEqual(params['face_smooth_intensity'], 0.0)

    def test_face_smoothing_with_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results)
        self.assertEqual(params['face_smooth_intensity'], 0.1)

    def test_face_smoothing_passport_with_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results, mode="passport")
        self.assertEqual(params['face_smooth_intensity'], 0.05) # Less aggressive

    def test_crop_no_face(self):
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_none, quality_results)
        expected_crop = [0, 0, self.image_dimensions[0], self.image_dimensions[1]]
        self.assertEqual(params['crop_rect'], expected_crop)

    def test_crop_one_face_general(self):
        # For this test, we expect the crop to center the face and apply padding.
        # Largest face: fx=100, fy=100, fw=200, fh=250. img_w=1000, img_h=800
        # crop_size = max(200, 250) * 1.8 = 250 * 1.8 = 450
        # crop_center_x = 100 + 200 / 2 = 200
        # crop_center_y = 100 + 250 * 0.4 = 100 + 100 = 200
        # crop_x = 200 - 450 / 2 = 200 - 225 = -25 -> adjusted to 0
        # crop_y = 200 - 450 / 2 = 200 - 225 = -25 -> adjusted to 0
        # crop_w = 450
        # crop_h = 450
        # Boundary checks:
        # crop_x = 0
        # crop_y = 0
        # crop_w = 450 (0 + 450 <= 1000)
        # crop_h = 450 (0 + 450 <= 800)
        # Enforce 1:1: min(450, 450) = 450. So, crop_w=450, crop_h=450
        # expected_crop = [0, 0, 450, 450]
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results)
        # Manual calculation based on logic:
        # Largest face: fx=100, fy=100, fw=200, fh=250
        # img_width=1000, img_height=800
        # crop_size = max(200, 250) * 1.8 = 450
        # crop_center_x = 100 + 200/2 = 200
        # crop_center_y = 100 + 250 * 0.4 = 200
        # crop_x = 200 - 450/2 = -25
        # crop_y = 200 - 450/2 = -25
        # crop_w, crop_h = 450, 450
        # Boundary adjustments:
        # crop_x = 0 if crop_x < 0 else crop_x -> 0
        # crop_y = 0 if crop_y < 0 else crop_y -> 0
        # crop_w = img_width - crop_x if crop_x + crop_w > img_width else crop_w -> 1000 - 0 = 1000 if -25 + 450 > 1000. No, 425 < 1000. So crop_w = 450
        # crop_h = img_height - crop_y if crop_y + crop_h > img_height else crop_h -> 800 - 0 = 800 if -25 + 450 > 800. No, 425 < 800. So crop_h = 450
        # current crop_x=0, crop_y=0, crop_w=450, crop_h=450
        # Enforce 1:1 aspect ratio
        # final_size = min(crop_w, crop_h) = min(450, 450) = 450
        # crop_w = 450
        # crop_h = 450
        # Re-center if size changed significantly and led to one side being cut more than other.
        # The logic is: cap x, y at 0. Then cap w,h based on img_dim - x/y. Then make square.
        # So expected is [0, 0, 450, 450]
        expected_crop = [0, 0, 450, 450]
        self.assertEqual(params['crop_rect'], expected_crop)


    def test_crop_largest_face_selected(self):
        # Ensures the largest face is used for cropping calculations
        # Face 1: [100,100,200,250], area 50000
        # Face 2: [500,300,300,350], area 105000 (selected)
        # fx=500, fy=300, fw=300, fh=350. img_w=1000, img_h=800
        # crop_size = max(300, 350) * 1.8 = 350 * 1.8 = 630
        # crop_center_x = 500 + 300 / 2 = 650
        # crop_center_y = 300 + 350 * 0.4 = 300 + 140 = 440
        # crop_x = 650 - 630 / 2 = 650 - 315 = 335
        # crop_y = 440 - 630 / 2 = 440 - 315 = 125
        # crop_w = 630
        # crop_h = 630
        # Boundary checks:
        # crop_x = 335 (>=0)
        # crop_y = 125 (>=0)
        # crop_w = 630 (335 + 630 = 965 <= 1000)
        # crop_h = 630 (125 + 630 = 755 <= 800)
        # Enforce 1:1: min(630, 630) = 630.
        # expected_crop = [335, 125, 630, 630]
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_two_faces, quality_results)
        expected_crop = [335, 125, 630, 630]
        self.assertEqual(params['crop_rect'], expected_crop)

    def test_crop_passport_mode_one_face(self):
        # Face: fx=100, fy=100, fw=200, fh=250. img_w=1000, img_h=800
        # crop_h_passport = fh / 0.6 = 250 / 0.6 = 416.66... -> 416
        # crop_w_passport = 416
        # crop_x_passport = (100 + 200/2) - 416/2 = 200 - 208 = -8 -> adjusted to 0
        # crop_y_passport = (100 + 250/2) - 416/2 = 225 - 208 = 17
        # Boundary checks:
        # crop_x = 0
        # crop_y = 17
        # crop_w = 416 (0 + 416 <= 1000)
        # crop_h = 416 (17 + 416 = 433 <= 800)
        # Enforce 1:1: min(416, 416) = 416
        # expected_crop = [0, 17, 416, 416]
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, self.mock_face_results_one_face, quality_results, mode="passport")
        # fh = 250
        # crop_h_passport = int(250 / 0.6) = 416
        # crop_w_passport = 416
        # crop_x_passport = int((100 + 200/2) - 416/2) = int(200 - 208) = -8
        # crop_y_passport = int((100 + 250/2) - 416/2) = int(100 + 125 - 208) = int(225 - 208) = 17
        # Adjustments:
        # crop_x = 0
        # crop_y = 17
        # crop_w = 416 (0+416 <= 1000)
        # crop_h = 416 (17+416 = 433 <= 800)
        # final_size = min(416,416) = 416.
        expected_crop = [0, 17, 416, 416]
        self.assertEqual(params['crop_rect'], expected_crop)

    def test_crop_boundary_conditions_width_height_capping(self):
        # Test case where initial crop would exceed image boundaries, forcing capping.
        # Face is large and near edge: fx=600, fy=500, fw=500, fh=400. img_w=1000, img_h=800
        # crop_size = max(500,400) * 1.8 = 500 * 1.8 = 900
        # crop_center_x = 600 + 500/2 = 850
        # crop_center_y = 500 + 400*0.4 = 500 + 160 = 660
        # crop_x = 850 - 900/2 = 850 - 450 = 400
        # crop_y = 660 - 900/2 = 660 - 450 = 210
        # Initial crop_w=900, crop_h=900
        # Boundary checks:
        # crop_x = 400 (>=0)
        # crop_y = 210 (>=0)
        # Check crop_x + crop_w: 400 + 900 = 1300. Exceeds img_width (1000).
        # crop_w_adjusted = 1000 - 400 = 600
        # Check crop_y + crop_h: 210 + 900 = 1110. Exceeds img_height (800).
        # crop_h_adjusted = 800 - 210 = 590
        # So, current crop_x=400, crop_y=210, crop_w=600, crop_h=590
        # Enforce 1:1 aspect ratio:
        # final_size = min(600, 590) = 590
        # crop_w = 590, crop_h = 590
        # expected_crop = [400, 210, 590, 590]
        face_near_edge = [{'box': [600, 500, 500, 400], 'confidence': 1.0}]
        quality_results = {'brightness': 0.5, 'contrast': 50}
        params = calculate_auto_enhancements(self.image_path, self.image_dimensions, face_near_edge, quality_results)
        expected_crop = [400, 210, 590, 590]
        self.assertEqual(params['crop_rect'], expected_crop)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
