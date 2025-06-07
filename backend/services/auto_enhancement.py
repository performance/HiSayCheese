from typing import Optional, List, Dict, Any # Added List, Dict, Any for more precise typing later if needed

def calculate_auto_enhancements(
    image_path: str,
    image_dimensions: tuple[int, int],
    face_detection_results: List[Dict[str, Any]], # Assuming list of dicts e.g. from FaceDetectionResponse
    image_quality_results: Dict[str, Any], # Assuming dict from ImageQualityAnalysisResponse
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates automatic image enhancement parameters based on analysis results.
    """
    # Extract brightness and contrast from image_quality_results
    # Default to 0.5 (neutral) for brightness and 50 for contrast if not found,
    # though they should ideally always be present.
    brightness = image_quality_results.get('brightness', 0.5)
    contrast = image_quality_results.get('contrast', 50.0) # Default to a mid-range value

    # Extract image dimensions
    img_width, img_height = image_dimensions

    # Brightness Adjustment Logic
    DARK_THRESHOLD = 0.3
    BRIGHT_THRESHOLD = 0.7
    brightness_target: float # Explicitly typing for clarity

    if mode == "passport":
        if brightness < 0.4:
            brightness_target = 1.2
        elif brightness > 0.6:
            brightness_target = 0.9
        else:
            brightness_target = 1.0
    else: # Default mode adjustments
        if brightness < DARK_THRESHOLD:
            brightness_target = 1.5
        elif brightness > BRIGHT_THRESHOLD:
            brightness_target = 0.8
        else:
            brightness_target = 1.1

    # Contrast Adjustment Logic
    LOW_CONTRAST_THRESHOLD = 20.0 # From image_quality.py, ensure float for comparison
    contrast_target: float # Explicitly typing for clarity

    if mode == "passport":
        if contrast < LOW_CONTRAST_THRESHOLD:
            contrast_target = 1.1
        else:
            contrast_target = 1.0
    else: # Default mode adjustments
        if contrast < LOW_CONTRAST_THRESHOLD:
            contrast_target = 1.2
        else:
            contrast_target = 1.05

    # Saturation Adjustment
    saturation_target: float
    if mode == "passport":
        saturation_target = 1.0
    else:
        saturation_target = 1.1

    # Background Blur
    background_blur_radius: int
    if mode == "passport":
        background_blur_radius = 0
    else:
        if face_detection_results: # Check if list is not empty
            background_blur_radius = 3
        else:
            background_blur_radius = 0

    # Face Smoothing
    face_smooth_intensity: float
    if mode == "passport":
        # For passport, apply mild smoothing if faces are detected, otherwise 0
        face_smooth_intensity = 0.05 if face_detection_results else 0.0
    else:
        if face_detection_results: # Check if list is not empty
            face_smooth_intensity = 0.1
        else:
            face_smooth_intensity = 0.0

    # Face Centering and Cropping Logic
    crop_rect = [0, 0, img_width, img_height] # Default to full image

    if face_detection_results:
        largest_face_box = None
        max_area = 0
        for face_data in face_detection_results:
            box = face_data.get('box') # Assuming box is [x, y, w, h]
            if box and len(box) == 4:
                fx, fy, fw, fh = box
                area = fw * fh
                if area > max_area:
                    max_area = area
                    largest_face_box = box

        if largest_face_box:
            fx, fy, fw, fh = largest_face_box

            crop_x, crop_y, crop_w, crop_h = 0.0, 0.0, 0.0, 0.0

            if mode == "passport":
                # Passport mode: face height should be ~60% of crop height, 1:1 aspect ratio
                crop_h_passport = fh / 0.6
                crop_w_passport = crop_h_passport # Maintain 1:1 aspect ratio

                # Center the face
                crop_x = (fx + fw / 2) - crop_w_passport / 2
                crop_y = (fy + fh / 2) - crop_h_passport / 2
                crop_w = crop_w_passport
                crop_h = crop_h_passport
            else:
                # General mode: Face in upper-center third, 1:1 aspect ratio
                crop_center_x = fx + fw / 2
                crop_center_y = fy + fh * 0.4  # Slightly above face center

                # Make crop window large enough to include face with padding
                crop_size = max(fw, fh) * 1.8

                crop_x = crop_center_x - crop_size / 2
                crop_y = crop_center_y - crop_size / 2
                crop_w = crop_size
                crop_h = crop_size

            # Boundary Checks and Adjustments
            # Ensure crop_x and crop_y are not negative
            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)

            # Ensure crop_x + crop_w is within image width
            if crop_x + crop_w > img_width:
                crop_w = img_width - crop_x

            # Ensure crop_y + crop_h is within image height
            if crop_y + crop_h > img_height:
                crop_h = img_height - crop_y

            # Enforce 1:1 aspect ratio after boundary adjustments by using the smaller dimension
            final_crop_size = min(crop_w, crop_h)
            # To keep it centered as much as possible after making it square
            # Re-calculate x/y based on the new final_crop_size if the original center point is preferred
            # For simplicity here, we adjust one of the coordinates if the center was shifted due to boundary capping.
            # This part could be more sophisticated to re-center based on the initial crop_center_x/y
            # However, for now, let's ensure it's square using the new size.
            # If crop_w was reduced, adjust crop_x to keep the face centered, if possible.
            # If crop_h was reduced, adjust crop_y.
            # This simple way just makes it square using top-left as anchor.
            crop_w = final_crop_size
            crop_h = final_crop_size

            # Re-check boundaries, although capping to min should prevent exceeding.
            # (This might be redundant if logic above is perfect, but safe)
            if crop_x + crop_w > img_width:
                crop_w = img_width - crop_x
            if crop_y + crop_h > img_height:
                crop_h = img_height - crop_y
            # And make it square again if the above changed anything (unlikely here)
            final_crop_size = min(crop_w, crop_h)
            crop_w = final_crop_size
            crop_h = final_crop_size


            crop_rect = [int(crop_x), int(crop_y), int(crop_w), int(crop_h)]

    # Return enhancement parameters
    return {
        "brightness_target": round(brightness_target, 2),
        "contrast_target": round(contrast_target, 2),
        "saturation_target": round(saturation_target, 2),
        "background_blur_radius": background_blur_radius,
        "crop_rect": crop_rect,
        "face_smooth_intensity": round(face_smooth_intensity, 2),
    }
