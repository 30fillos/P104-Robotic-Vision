
import cv2


def convert_to_grayscale(image):

    # Convert BGR image to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image
