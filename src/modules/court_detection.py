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


def adjust_pt(pt, remaining_distance, axis, direction):
    if axis == 'x':
        pt[0] += int(direction * remaining_distance)
    elif axis == 'y':
        pt[1] += int(direction * remaining_distance)
    return pt

    
def find_service_box_center_point(img):
    for white_threshold in range(150, 90, -10):
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (5, 5), 0)

            # Threshold for white pixels
            lower_white = np.array([white_threshold, white_threshold, white_threshold])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(blurred, lower_white, upper_white)
            
            # Apply HoughLinesP
            linesP = cv2.HoughLinesP(white_mask, 
                                    rho=1, 
                                    theta=np.pi/180, 
                                    threshold=30, 
                                    minLineLength=30, 
                                    maxLineGap=10)
            
            # Filter for vertical and centered lines
            vert_cent_linesP = filter_center_linesP(filter_vertical_linesP(linesP), white_mask.shape[1])
            center_line_mask = ip.draw_lines(np.zeros((256,256)), vert_cent_linesP)

            # Find the center point of the service box
            center_line_white_mask = np.where(white_mask, center_line_mask, 0)
            
            # Find the y, x coordinates of all non-zero pixels
            non_zero_y, non_zero_x = np.nonzero(center_line_white_mask)
            index = np.argmax(non_zero_y) # Find the index of the bottom-most point

            return non_zero_x[index], non_zero_y[index]
        except:
            print(f"Retrying with white_threshold={white_threshold-10}")


def find_service_box_corner_points(img, center_x, center_y):
    for white_threshold in range(150, 90, -10):
        try: 
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (3, 3), 0)

            # Create a horizontal line mask based on the center point
            mask = np.zeros_like(blurred)
            mask[center_y-7:center_y+7, :] = 255

            # Prepare image to only show service line
            service_line_img = np.where(mask, blurred, 0)
            
            # Threshold for white pixels
            lower_white = np.array([white_threshold, white_threshold, white_threshold])
            upper_white = np.array([255, 255, 255])
            white_service_line = cv2.inRange(service_line_img, lower_white, upper_white)

            # Use horizontal kernel to detect service line
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            service_line = cv2.morphologyEx(white_service_line, cv2.MORPH_OPEN, h_kernel)

            # Apply HoughlinesP
            linesP = cv2.HoughLinesP(service_line,
                                    rho=1,
                                    theta=np.pi/180,
                                    threshold=50,
                                    minLineLength=30,
                                    maxLineGap=30)
            service_line_mask = ip.draw_lines(np.zeros((256,256)), linesP, line_thickness=2)

            # Check left endpoint and right endpoint are approximately the same distance
            # from the center point
            non_zero_y, non_zero_x = np.nonzero(service_line_mask)
            left_pt_idx = np.argmin(non_zero_x)
            right_pt_idx = np.argmax(non_zero_x)
            left_pt = [non_zero_x[left_pt_idx], non_zero_y[left_pt_idx]]
            right_pt = [non_zero_x[right_pt_idx], non_zero_y[right_pt_idx]]
            center_pt = [center_x, center_y]

            # Draw points on the image
            # dots_img = np.copy(img)
            # cv2.circle(dots_img, center_pt, 3, (255, 0, 0), -1)
            # cv2.circle(dots_img, right_pt, 3, (255, 0, 0), -1)
            # cv2.circle(dots_img, left_pt, 3, (255, 0, 0), -1)
            # ip.display_image(dots_img, title="initial dots")

            # Distances from center point
            left_distance = np.linalg.norm(np.array(left_pt) - np.array(center_pt))
            right_distance = np.linalg.norm(np.array(right_pt) - np.array(center_pt))
            # Adjust the points to be approximately the same distance from the center point
            if left_distance < right_distance - 4:
                adjust_pt(left_pt, right_distance - left_distance, 'x', -1)
            elif right_distance < left_distance - 4:
                adjust_pt(right_pt, left_distance - right_distance, 'x', 1)

            # cv2.circle(dots_img, center_pt, 3, (0, 255, 0), -1)
            # cv2.circle(dots_img, right_pt, 3, (0, 255, 0), -1)
            # cv2.circle(dots_img, left_pt, 3, (0, 255, 0), -1)
            # ip.display_image(dots_img, title="continuation of dots")
            
            return left_pt, right_pt
        except:
            print(f"Retrying with white_threshold={white_threshold-10}")




def main():
    # Read the image with color
    # gray_image = ip.read_image('data/images/example1_0.png')
    image = ip.read_image('data/images/example2_0.png', cv2.COLOR_BGR2RGB)
    ip.display_image(image, False, 'Original Image')   

    # Detect center service box point
    center_pt = find_service_box_center_point(image)

    # Use center service box point to find left and right corners of service box
    left_pt, right_pt = find_service_box_corner_points(image, center_pt[0], center_pt[1])

    

    # Draw points on the image
    image_copy = np.copy(image)
    cv2.circle(image_copy, center_pt, 4, (255, 0, 0), -1)
    cv2.circle(image_copy, right_pt, 4, (255, 0, 0), -1)
    cv2.circle(image_copy, left_pt, 4, (255, 0, 0), -1)
    ip.display_image(image_copy, False, 'Final Image')

    # Final Step
    # Homography to transform image to top-down view


main()