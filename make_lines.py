import cv2
import numpy as np


def make_lines(image: str):
    """
    Process the text image to convert to lines
    """
    # Load image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection
    horizontal_projection = np.sum(binary, axis=1)

    # Detecting lines
    line_start = -1
    line_end = -1
    outputs = []
    for i, pixel_value in enumerate(horizontal_projection):
        if pixel_value > 0 and line_start < 0:
            line_start = i
        elif pixel_value <= 0 and line_start >= 0:
            line_end = i
            outputs.append(image[line_start:line_end, :].copy())
            line_start = -1

    if line_start >= 0:
        outputs.append(image[line_start:, :].copy())
    # Show result
    for output in outputs:
        cv2.imshow("Lines", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


path = "./try2.jpg"
img = cv2.imread(path)
make_lines(img)
