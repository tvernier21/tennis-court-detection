import image_processing as ip
import numpy as np
import cv2


def twoPoints2Polar(line):
    # Get points from the line
    p1 = np.array([line[0], line[1]])
    p2 = np.array([line[2], line[3]])

    # Compute 'rho' and 'theta'
    rho = abs(p2[0]*p1[1] - p2[1]*p1[0]) / np.linalg.norm(p2 - p1)
    theta = -np.arctan2((p2[0] - p1[0]) , (p2[1] - p1[1]))

    # You can have a negative distance from the center 
    # when the angle is negative
    if theta < 0:
        rho = -rho

    return rho, theta


def convert_houghlinesP_to_houghlines(lines_p):
    hough_lines = []
    if lines_p is not None:
        for line in lines_p:
            rho, theta = twoPoints2Polar(line)
            hough_lines.append([rho,theta])

    out = np.array(hough_lines)
    return out.reshape(out.shape[0], 1, out.shape[1])


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


# def detect_lines(image):
#     # Canny edge detection -> Hough Line transformation
#     edges = cv2.Canny(image, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 60) # Low filter (50)

#     vlines = filter_vertical_lines(lines)
#     vlines_center = filter_center_lines(vlines, image.shape[1])

#     # create a mask using the drawn lines
#     vc_lines_mask = ip.draw_houghlines(np.zeros((256,256)), vlines_center)
#     img_vc_mask = np.where(vc_lines_mask, image, 0)
#     ip.display_image(img_vc_mask)

#     hlines = filter_horizontal_lines(lines)
#     h_lines_mask = ip.draw_houghlines(np.zeros((256,256)), hlines)
#     img_h_mask = np.where(h_lines_mask, image, 0)
#     ip.display_image(img_h_mask)

#     return "poop"


# def init_court_detection(image):
#     # Process using Gaussian Blur
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     ip.display_image(blurred, title = 'Blurred Image')

#     adaptive = cv2.adaptiveThreshold(blurred, 
#                                     255, 
#                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY, 
#                                     11, 
#                                     2)
#     ip.display_image(adaptive, title = 'Blurred Adaptive Threshold')

#     adaptive = cv2.adaptiveThreshold(image, 
#                                     255, 
#                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                     cv2.THRESH_BINARY, 
#                                     11, 
#                                     2)
#     ip.display_image(adaptive, title = ' Adaptive Threshold')

#     # Process using Canny Edge Detection
#     edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
#     ip.display_image(edges, title = 'Canny Edge Detection')

    
def find_white_lines(img):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Define range for white color
    lower_white = np.array([130, 130, 130])
    upper_white = np.array([255, 255, 255])

    # Threshold the image to get only white colors
    white_mask = cv2.inRange(blurred, lower_white, upper_white)

    ip.display_image(white_mask)
    
    # Apply HoughLinesP
    linesP = cv2.HoughLinesP(white_mask, rho=1, theta=np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)

    vc_linesP = filter_center_lines(filter_vertical_lines(linesP), img.shape[1])
    v_linesP = filter_vertical_lines(linesP)

    img2 = ip.draw_houghlines(img, vc_linesP)
    ip.display_image(img2)

    return



def main():
    # Read the image with color
    gray_image = ip.read_image('data/images/example1_0.png')
    image = ip.read_image('data/images/example1_0.png', cv2.COLOR_BGR2RGB)
    # ip.display_image(image, False, 'Original Image')   

    # First detect_lines
    # init_court_detection(image)
    find_white_lines(image)


main()