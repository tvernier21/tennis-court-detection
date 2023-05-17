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


def filter_vertical_linesP(lines, min_slope=3):
    # Find vertical lines based on their slope
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x = (x2 - x1)
        if x == 0: x = .001

        if abs((y2 - y1) / x) > min_slope:
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


def filter_center_linesP(lines, image_width):
    # Filter for lines only in the center of the image (.4, .6)
    filtered_lines = []
    for line in lines:
        x1, _, x2, _ = line[0]
        x_avg = (x1 + x2) / 2
        if x_avg > image_width * .3 and x_avg < image_width * .7:
            filtered_lines.append(line)
    return filtered_lines

    
def find_service_box_center_point(img):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Define range for white color
    lower_white = np.array([130, 130, 130])
    upper_white = np.array([255, 255, 255])

    # Threshold the image to get only white colors
    white_mask = cv2.inRange(blurred, lower_white, upper_white)

    # ip.display_image(white_mask)
    
    # Apply HoughLinesP
    linesP = cv2.HoughLinesP(white_mask, 
                             rho=1, 
                             theta=np.pi/180, 
                             threshold=30, 
                             minLineLength=30, 
                             maxLineGap=10)
    
    # Find vertical lines and filter for lines in the center of the image
    vc_linesP = filter_center_linesP(filter_vertical_linesP(linesP), white_mask.shape[1])
    center_line_mask = ip.draw_lines(np.zeros((256,256)), vc_linesP)
    # ip.display_image(center_line_mask, title="Center Line w/ HoughLinesP")

    # Find the center point of the service box
    center_line_white_mask = np.where(white_mask, center_line_mask, 0)
    
    # Find the y, x coordinates of all non-zero pixels
    non_zero_y, non_zero_x = np.nonzero(center_line_white_mask)
    # Find the index of the non-zero pixel with the largest y coordinate
    index = np.argmax(non_zero_y)

    return non_zero_x[index], non_zero_y[index]


def find_service_box_edge_points(img, center_x, center_y):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Create a horizontal line mask based on the center point
    mask = np.zeros_like(blurred)
    mask[center_y-7:center_y+7, :] = 255

    # Apply the mask to the image
    service_line_img = np.where(mask, blurred, 0)

    ip.display_image(service_line_img, title="Service Line Image")
    
    # Threshold to filter for white pixels
     # Define range for white color
    lower_white = np.array([130, 130, 130])
    upper_white = np.array([255, 255, 255])
    white_service_line = cv2.inRange(service_line_img, lower_white, upper_white)
    ip.display_image(white_service_line, title="White Service Line")

    # Define a horizontal line kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))

    # Apply morphological operations to filter for white pixels aligned on a horizontal line
    processed_img = cv2.morphologyEx(white_service_line, cv2.MORPH_OPEN, kernel)

    ip.display_image(processed_img, title="Processed Image")




def main():
    # Read the image with color
    # gray_image = ip.read_image('data/images/example1_0.png')
    image = ip.read_image('data/images/example1_0.png', cv2.COLOR_BGR2RGB)
    # ip.display_image(image, False, 'Original Image')   

    # Detect center service box point
    center_x, center_y = find_service_box_center_point(image)
    # image_copy = np.copy(image)
    # cv2.circle(image_copy, (center_x, center_y), 3, (255, 0, 0), -1)
    # ip.display_image(image_copy, False, 'Center Point')

    # Use center service box point to find left and right corners of service box
    find_service_box_edge_points(image, center_x, center_y)


main()