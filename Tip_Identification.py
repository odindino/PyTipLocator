import cv2
import numpy as np


def detect_tip_and_reflection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=50, minLineLength=50, maxLineGap=10)

    # Find the two longest lines (tip and reflection)
    if lines is not None:
        sorted_lines = sorted(lines, key=lambda x: np.linalg.norm(
            x[0][:2]-x[0][2:]), reverse=True)
        tip_line = sorted_lines[0][0]
        reflection_line = sorted_lines[1][0]
        return tip_line, reflection_line
    return None, None


def calculate_symmetry_line(tip_line, reflection_line):
    # Calculate midpoints of the lines
    tip_midpoint = ((tip_line[0] + tip_line[2]) // 2,
                    (tip_line[1] + tip_line[3]) // 2)
    reflection_midpoint = (
        (reflection_line[0] + reflection_line[2]) // 2, (reflection_line[1] + reflection_line[3]) // 2)

    # Calculate symmetry line
    symmetry_line = (tip_midpoint, reflection_midpoint)
    return symmetry_line


def project_tip_to_symmetry_line(tip_line, symmetry_line):
    # Assuming the tip is at the end of the tip_line closer to the symmetry_line
    tip_point = (tip_line[2], tip_line[3])  # End point of the tip line

    # Calculate the direction vector of the symmetry line
    symmetry_vector = (symmetry_line[1][0] - symmetry_line[0]
                       [0], symmetry_line[1][1] - symmetry_line[0][1])

    # Calculate the projection
    t = ((tip_point[0] - symmetry_line[0][0]) * symmetry_vector[0] +
         (tip_point[1] - symmetry_line[0][1]) * symmetry_vector[1]) / (symmetry_vector[0]**2 + symmetry_vector[1]**2)

    projected_point = (int(symmetry_line[0][0] + t * symmetry_vector[0]),
                       int(symmetry_line[0][1] + t * symmetry_vector[1]))

    return projected_point


def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Detect tip and reflection
    tip_line, reflection_line = detect_tip_and_reflection(image)

    if tip_line is not None and reflection_line is not None:
        # Calculate symmetry line
        symmetry_line = calculate_symmetry_line(tip_line, reflection_line)

        # Project tip to symmetry line
        contact_point = project_tip_to_symmetry_line(tip_line, symmetry_line)

        # Draw lines and points
        cv2.line(image, (tip_line[0], tip_line[1]),
                 (tip_line[2], tip_line[3]), (0, 255, 0), 2)
        cv2.line(image, (reflection_line[0], reflection_line[1]),
                 (reflection_line[2], reflection_line[3]), (0, 255, 0), 2)
        cv2.line(image, symmetry_line[0], symmetry_line[1], (255, 0, 0), 2)
        cv2.circle(image, contact_point, 5, (0, 0, 255), -1)

        # Display the result
        cv2.imshow('Processed Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not detect tip and reflection.")


# Usage
process_image('Tip locate_test5.png')
