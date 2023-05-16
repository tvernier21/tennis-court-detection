import image_processing as ip
import numpy as np
import cv2


def filter_vertical_lines(lines, theta_threshold=.3):
    # Filter lines based on their angle to the horizontal axis
    filtered_lines = []
    for line in lines:
        theta = line[0][1]
        if theta < theta_threshold or theta > np.pi - theta_threshold:
            filtered_lines.append(line)
    return filtered_lines


def filter_horizontal_lines(lines, theta_threshold=.2):
    filtered_lines = []
    for line in lines:
        theta = line[0][1]
        if theta < (np.pi / 2) + theta_threshold and \
            theta > (np.pi / 2) - theta_threshold:
            filtered_lines.append(line)
    return filtered_lines


def filter_center_lines(lines, image_width):
    # Filter for lines only in the center of the image (.4, .6)
    filtered_lines = []
    for line in lines:
        rho = line[0][0]
        if abs(rho) > image_width * .3 and abs(rho) < image_width * .7:
            filtered_lines.append(line)
    return filtered_lines


def detect_lines(image):
    # Canny edge detection -> Hough Line transformation
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60) # Low filter (50)

    vlines = filter_vertical_lines(lines)
    vlines_center = filter_center_lines(vlines, image.shape[1])

    # create a mask using the drawn lines
    vc_lines_mask = ip.draw_houghlines(np.zeros((256,256)), vlines_center)
    img_vc_mask = np.where(vc_lines_mask, image, 0)
    ip.display_image(img_vc_mask)

    hlines = filter_horizontal_lines(lines)
    h_lines_mask = ip.draw_houghlines(np.zeros((256,256)), hlines)
    img_h_mask = np.where(h_lines_mask, image, 0)
    ip.display_image(img_h_mask)

    return "poop"

def init_court_detection(image):
    # Process using Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ip.display_image(blurred, 'Blurred Image')

    bgak = cv2.adaptiveThreshold(blurred, 
                                 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 
                                 11, 
                                 2)
    ip.display_image(bgak, 'Blurred Adaptive Threshold')

    # Process using Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    ip.display_image(edges, 'Canny Edge Detection')

    # process using bilateral filter
    bilateral = cv2.bilateralFilter(image, 30, 25, 75)
    ip.display_image(bilateral, 'Bilateral Filter')



def main():
    # Read the image with color
    image = ip.read_image('data/images/example1_0.png')
    # ip.display_image(image, False, 'Original Image')   

    # First detect_lines
    init_court_detection(image)

main()