import cv2
import numpy as np


def preprocess_image(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)

    return denoised


def detect_tip_and_reflection(image):
    # Apply threshold to separate dark regions (potential tips) from bright background
    _, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to enhance tip shape
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on aspect ratio and area
    tip_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w if w > 0 else 0
        area = cv2.contourArea(cnt)
        if aspect_ratio > 2 and area > 100:  # Adjust these thresholds as needed
            tip_candidates.append(cnt)

    # Sort candidates by y-coordinate
    tip_candidates.sort(key=lambda c: cv2.boundingRect(c)[1])

    if len(tip_candidates) >= 2:
        # Return topmost and bottommost candidates
        return tip_candidates[0], tip_candidates[-1]
    elif len(tip_candidates) == 1:
        return tip_candidates[0], None
    else:
        return None, None


def calculate_symmetry_line(tip, reflection):
    if reflection is None:
        return None

    tip_center = np.mean(tip.squeeze(), axis=0)
    reflection_center = np.mean(reflection.squeeze(), axis=0)

    mid_point = ((tip_center[0] + reflection_center[0]) // 2,
                 (tip_center[1] + reflection_center[1]) // 2)

    return (mid_point, (mid_point[0], image.shape[0]))


def process_image(image_path):
    global image  # Make image global so it can be used in other functions if needed
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    preprocessed = preprocess_image(image)
    tip, reflection = detect_tip_and_reflection(preprocessed)

    if tip is not None:
        # Create a color image for visualization
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw tip contour
        cv2.drawContours(color_image, [tip], 0, (0, 255, 0), 2)

        if reflection is not None:
            # Draw reflection contour
            cv2.drawContours(color_image, [reflection], 0, (0, 255, 0), 2)

            # Calculate and draw symmetry line
            symmetry_line = calculate_symmetry_line(tip, reflection)
            if symmetry_line:
                cv2.line(
                    color_image, symmetry_line[0], symmetry_line[1], (255, 0, 0), 2)

        # Display the result
        cv2.imshow('Processed Image', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not detect tip and reflection.")


# Usage
process_image('Tip locate_test5.png')
