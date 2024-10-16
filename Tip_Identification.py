import cv2
import numpy as np
import os


def save_image(image, name):
    cv2.imwrite(f"{name}.jpg", image)
    cv2.imshow(name, image)
    cv2.waitKey(0)


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, "1_grayscale")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_image(blurred, "2_blurred")

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    save_image(thresh, "3_threshold")

    return thresh


def morphological_operations(binary):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    save_image(opening, "4_opening")

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    save_image(closing, "5_closing")

    return closing


def detect_tip_and_reflection(processed, original_image):
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    save_image(contour_image, "6_all_contours")

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 0.1 < aspect_ratio < 10:  # Filter based on aspect ratio
                valid_contours.append(cnt)

    filtered_contour_image = original_image.copy()
    cv2.drawContours(filtered_contour_image,
                     valid_contours, -1, (0, 255, 0), 2)
    save_image(filtered_contour_image, "7_filtered_contours")

    if len(valid_contours) < 2:
        print(f"Not enough valid contours. Found: {len(valid_contours)}")
        return None, None

    # Sort contours by y-coordinate
    sorted_contours = sorted(
        valid_contours, key=lambda c: cv2.boundingRect(c)[1])

    tip = sorted_contours[0]
    reflection = sorted_contours[-1]

    return tip, reflection


def calculate_symmetry_line(tip, reflection, image):
    tip_M = cv2.moments(tip)
    reflection_M = cv2.moments(reflection)

    tip_cx, tip_cy = int(tip_M['m10']/tip_M['m00']
                         ), int(tip_M['m01']/tip_M['m00'])
    reflection_cx, reflection_cy = int(
        reflection_M['m10']/reflection_M['m00']), int(reflection_M['m01']/reflection_M['m00'])

    mid_x = (tip_cx + reflection_cx) // 2

    return ((mid_x, 0), (mid_x, image.shape[0]))


def process_image(image_path):
    image = cv2.imread(image_path)
    save_image(image, "0_original")

    processed = preprocess_image(image)
    morphed = morphological_operations(processed)

    tip, reflection = detect_tip_and_reflection(morphed, image)

    if tip is not None and reflection is not None:
        result_image = image.copy()
        cv2.drawContours(result_image, [tip], 0, (0, 255, 0), 2)
        cv2.drawContours(result_image, [reflection], 0, (0, 255, 0), 2)

        symmetry_line = calculate_symmetry_line(tip, reflection, image)
        cv2.line(result_image, symmetry_line[0],
                 symmetry_line[1], (255, 0, 0), 2)

        # Calculate contact point (tip of the tip contour)
        tip_top = tuple(tip[tip[:, :, 1].argmin()][0])
        cv2.circle(result_image, tip_top, 5, (0, 0, 255), -1)

        save_image(result_image, "8_final_result")
    else:
        print("Could not detect tip and reflection.")

    cv2.destroyAllWindows()


# Usage
process_image('Tip locate_test5.png')
