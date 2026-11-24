
from pathlib import Path
import cv2


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the images directory
DATA_DIR = PROJECT_ROOT / "data" / "image_analysis" / "images"


def load_image(image_name):

    # Create the full image path
    image_path = DATA_DIR / image_name

    # Check if the image exists
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return None

    # Load the image using OpenCV
    image = cv2.imread(str(image_path))

    # Check if OpenCV successfully loaded the image
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return None

    print(f"Image loaded successfully: {image_name}")
    print(f"Dimensions: {image.shape}")

    return image
