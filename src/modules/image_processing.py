import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image, gray=True, title='Image'):
    # Use Matplotlib to display the image
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # Hide the x and y axis
    plt.show()


def read_image(image_path, colorfilter=cv2.COLOR_BGR2GRAY):
    # Read the image and filter using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, colorfilter)
    return image


def draw_lines(image, lines):
    # copy the passed image
    image_copy = np.copy(image)

    # Iterate over all detected lines and draw them on the copy of the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_copy, (x1, y1), (x2, y2), 255, 2)

    return image_copy


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
        cv2.line(image_copy, (x1, y1), (x2, y2), 255, 2)

    # Return the modified image
    return image_copy