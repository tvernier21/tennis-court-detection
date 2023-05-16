import image_processing as ip
import numpy as np
import cv2


def filter_vertical_lines(lines, theta_threshold=0.3):
    # Filter lines based on their angle to the horizontal axis
    filtered_lines = []
    for line in lines:
        theta = line[0][1]
        if theta < theta_threshold or theta > np.pi - theta_threshold:
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
    ip.display_image(ip.draw_houghlines(image, vlines_center))

    # image_lines = ip.draw_houghlines(image, lines)
    # ip.display_image(image_lines)


def main():
    # Read the image
    image = ip.read_image('data/images/example1_0.png')

    # First detect_lines
    detect_lines(image)

main()