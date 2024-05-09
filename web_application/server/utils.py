import cv2
import numpy as np


def crop_width(real_img):
    img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Vertical projection
    vertical_projection = np.nonzero(np.sum(binary, axis=0))[0]
    if len(vertical_projection) == 0:
        return real_img
    return real_img[:, vertical_projection[0]:vertical_projection[-1] + 1]

def make_lines(path: str=None, img=None):
    """
    Process the text image to convert to lines
    """
    if path is not None:
        # Read the image
        real_image = cv2.imread(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        real_image = img
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)

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
            cropped = crop_width(real_image[line_start:line_end, :].copy())
            outputs.append(cropped)
            line_start = -1

    if line_start >= 0:
        outputs.append(real_image[line_start:, :].copy())
    print(f"Detected {len(outputs)} lines")
    
    return outputs

def summarize_image(ocr_model, summarizer_model, min_length=100, max_length=150, img_path: str=None, img=None):
    """
    Predict the text from the image
    """
    assert img_path is not None or img is not None, "Either img_path or img should be provided"
    assert img_path is None or img is None, "Either img_path or img should be provided, not both"

    lines = make_lines(img_path, img)
    if len(lines) == 0:
        raise ValueError("No lines detected in the image")
    text = []
    for line in lines:
        text.append(ocr_model.generate(line))
    text = " ".join(text)
    summary = summarizer_model.summarize(text, min_length=min_length, max_length=max_length)
    return summary

def recognize_image(ocr_model, img_path: str=None, img=None):
    """
    Predict the text from the image
    """
    assert img_path is not None or img is not None, "Either img_path or img should be provided"
    assert img_path is None or img is None, "Either img_path or img should be provided, not both"

    lines = make_lines(img_path, img)
    if len(lines) == 0:
        raise ValueError("No lines detected in the image")
    text = []
    for line in lines:
        text.append(ocr_model.generate(line))
    return " ".join(text)