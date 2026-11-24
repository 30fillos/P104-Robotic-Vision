
import cv2
from src.image_analysis.preprocessing import convert_to_grayscale
from src.image_analysis.image_loader import load_image


def main():

    print("================================")
    print("       ROBOTIC VISION")
    print("================================")

    # 1. Load image
    image = load_image("1.jpg")

    # 2. Check image loading
    if image is None:
        print("Image loading failed!")
        return

    
    # Convert image to Grayscale
    gray_image = convert_to_grayscale(image)

    print("\nGrayscale information:")
    print("Shape:", gray_image.shape)
    print("Data type:", gray_image.dtype)
    print("Total pixels:", gray_image.size)

   
    # 3. Image information
    print("\nImage information:")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Total pixels:", image.size)

    # 4. Pixel analysis
    print("\nPixel analysis:")

    pixel = image[100, 200]

    print("Pixel BGR values:", pixel)
    print("Blue:", pixel[0])
    print("Green:", pixel[1])
    print("Red:", pixel[2])

    # 5. Display image
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image", 1000, 700)
    cv2.imshow("Original Image", image)

    # 6. Wait for keyboard input
    cv2.waitKey(0)
 # Display Grayscale image
    cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Image", 1000, 700)
    cv2.imshow("Grayscale Image", gray_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 7. Close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


