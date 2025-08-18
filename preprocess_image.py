import cv2
import numpy as np
import os
from rembg import remove, new_session

def create_color_distance_mask(image: np.ndarray) -> np.ndarray:
    """Creates a high-contrast mask based on the color distance to white."""
    white_color = np.array([255, 255, 255])
    image_float = image.astype(np.float32)
    distance = np.sqrt(np.sum((image_float - white_color)**2, axis=2))
    normalized_distance = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.bitwise_not(normalized_distance)

def order_points(pts):
    """Sorts 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_and_undistort_paper(image:np.array, dpi: int = 100):
    """Paper detection and perspective correction (unchanged)."""
    A4_SHORT_MM, A4_LONG_MM, INCH_TO_MM = 210, 297, 25.4
    short_px, long_px = int((A4_SHORT_MM / INCH_TO_MM) * dpi), int((A4_LONG_MM / INCH_TO_MM) * dpi)

    #image = cv2.imread(image_path)
    if image is None: return None, None
    h_orig, w_orig = image.shape[:2]

    high_contrast_mask = create_color_distance_mask(image)
    _, final_thresh = cv2.threshold(high_contrast_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    main_contour = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(main_contour)
    perimeter = cv2.arcLength(hull, True)
    approx_corners = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(approx_corners) != 4: return None, None
    ordered_corners = order_points(approx_corners)

    width_a = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    width_b = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    target_paper_dims = (long_px, short_px) if max(width_a, width_b) > max(np.linalg.norm(ordered_corners[1] - ordered_corners[2]), np.linalg.norm(ordered_corners[0] - ordered_corners[3])) else (short_px, long_px)

    dst_paper_pts = np.array([[0, 0], [target_paper_dims[0] - 1, 0], [target_paper_dims[0] - 1, target_paper_dims[1] - 1], [0, target_paper_dims[1] - 1]], dtype='float32')
    paper_transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_paper_pts)

    img_corners = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype=np.float32)
    transformed_img_corners = cv2.perspectiveTransform(img_corners[np.newaxis, :, :], paper_transform_matrix)
    x_min, y_min, w, h = cv2.boundingRect(transformed_img_corners)

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    final_transform_matrix = translation_matrix.dot(paper_transform_matrix)
    final_size = (w, h)

    warped_image = cv2.warpPerspective(image, final_transform_matrix, final_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    final_paper_corners = cv2.perspectiveTransform(ordered_corners[np.newaxis, :, :], final_transform_matrix)

    return warped_image, final_paper_corners.reshape(4, 2)


def create_binary_mask(inputImage: np.ndarray) -> np.ndarray:
    #inputImageCopy = inputImage.copy()

    # Convert to float and divide by 255:
    imgFloat = inputImage.astype(float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255*kChannel).astype(np.uint8)
    _, binaryImage = cv2.threshold(kChannel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 5
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    return binaryImage


def crop_to_content(binary_mask: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Crops the image to the content area and returns the bounding box.
    """
    # Invert binary mask to find non-white pixels
    inv_binary_mask = cv2.bitwise_not(binary_mask)
    coords = cv2.findNonZero(inv_binary_mask)
    if coords is None:
        # No content found, return None for both image and bounding box
        return None, None
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = binary_mask[y:y+h, x:x+w]
    bbox = (x, y, w, h)
    return cropped_image, bbox

import cv2
import numpy as np

def dilate_contours(image, offset):
    if offset <= 0:
        return image.copy()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*offset + 1, 2*offset + 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY_INV)[1]
    return result

def add_white_border_pad(image, border_width):
    return np.pad(image, border_width, mode='constant', constant_values=255)


if __name__ == "__main__":
    INPUT_IMAGE = "test_img.jpg"
    OUTPUT_RECTIFIED = "A4_rectified_lightweight.jpg"
    OUTPUT_OUTLINES = "outlines_only.jpg"
    OUTPUT_TOOL = "tool_only.jpg"
    OUTPUT_CROPPED = "cropped_image.jpg"


    if not os.path.exists(INPUT_IMAGE):
        print("🖼️ Creating test image...")
        height, width = 1080, 1920
        background = np.full((height, width, 3), (50, 45, 42), dtype=np.uint8)

        # Paper
        paper_pts = np.array([[400, 200], [1450, 250], [1350, 900], [300, 850]], dtype=np.int32)
        cv2.fillConvexPoly(background, paper_pts, (250, 250, 250))

        # Various Tools
        # Large Screwdriver
        cv2.rectangle(background, (600, 50), (630, 950), (85, 70, 55), -1)
        cv2.rectangle(background, (620, 100), (640, 200), (120, 95, 75), -1)

        # Hammer
        cv2.rectangle(background, (800, 300), (830, 800), (90, 75, 60), -1)
        cv2.rectangle(background, (780, 350), (880, 420), (70, 60, 45), -1)

        # Pliers (more complex shape)
        pts1 = np.array([[1000, 400], [1020, 380], [1080, 420], [1100, 500],
                        [1050, 520], [1020, 480]], np.int32)
        cv2.fillPoly(background, [pts1], (75, 65, 50))

        cv2.imwrite(INPUT_IMAGE, background)
        print("✅ Test image created")

    # Main processing
    print("\n" + "="*50)
    print("🖼️ Input image:", INPUT_IMAGE)
    rectified_image, paper_coords = process_and_undistort_paper(cv2.imread(INPUT_IMAGE))

    if rectified_image is not None and paper_coords is not None:
        print(f"✅ Image rectified → '{OUTPUT_RECTIFIED}'")
        cv2.imwrite(OUTPUT_RECTIFIED, rectified_image)
        print("removing background...")
        #session=new_session("silueta")
        #removed_bg = remove(rectified_image,session=session)
        removed_bg = remove(rectified_image)
        print(f"✅ Background removed → '{OUTPUT_TOOL}'")
        cv2.imwrite(OUTPUT_TOOL, removed_bg)
        print("Creating floodfill mask...")
        bin_mask = create_binary_mask(removed_bg)
        print(f"✅ Floodfill mask created → '{OUTPUT_OUTLINES}'")
        cv2.imwrite(OUTPUT_OUTLINES, bin_mask)
        dilated_image = dilate_contours(bin_mask, 5)
        print("✅ Contours dilated")
        cv2.imwrite("dilated_contours.jpg", dilated_image)
        cropped_image, _ = crop_to_content(dilated_image) # Update call to unpack tuple
        if cropped_image is not None:
            padded_image = add_white_border_pad(cropped_image, 10)
            print("✅ White border added")
            cv2.imwrite("padded_image.jpg", padded_image)
            print("✅ Image cropped")
            cv2.imwrite(OUTPUT_CROPPED, cropped_image)
            print("Contours dilated and saved")
        else:
            print("❌ Error cropping the image")
        print("✅ Processing finished")


    else:
        print("❌ Processing failed")


#TODO:
#dilate contours
