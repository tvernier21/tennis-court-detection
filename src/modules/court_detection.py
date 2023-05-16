import image_preprocessing as ip
import cv2

def detect_lines(image):
    # Apply edge detection
    edges = cv2.Canny(gray_image, 50, 200)

    # Plot the result
    ip.display_image(edges)


def detect_lines2(image):
    # Perform edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform a Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    ip.draw_houghlines(image, lines)


def main():
    # Read the image
    image = ip.read_image('data/images/example1_0.png')
    ip.display_image(image)

    # First detect_lines
    detect_lines(image)
    detect_lines2(image)

main()