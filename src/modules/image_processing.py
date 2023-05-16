import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image):
    # Use Matplotlib to display the image
    plt.imshow(image)
    plt.axis('off')  # Hide the x and y axis
    plt.show()


def read_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # OpenCV reads images in BGR format, we'll convert it to RGB for displaying
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def draw_houghlines(image, lines):
    # Copy the passed image
    image_copy = np.copy(image)

    # Iterate over all detected lines and draw them on the copy of the image
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # We'll use 1000 as an arbitrary value for the length of the lines to be drawn
        scale = image.shape[0] + image.shape[1]

        # Compute the start and end points of the lines to be drawn
        x1 = int(x0 + scale * -b)
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * -b)
        y2 = int(y0 - scale * a)

        # Draw the lines on the image
        cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Return the modified image
    return image_copy