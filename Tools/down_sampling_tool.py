import os
import cv2

# -------------------------------
# Folder paths
# -------------------------------
input_dir = "../HR_ground_truth"
output_dir = "../LR"

# Create LR folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Check input folder
if not os.path.isdir(input_dir):
    print(f"Input folder not found: {os.path.abspath(input_dir)}")
    exit()

# Process every image
for filename in os.listdir(input_dir):

    if not filename.lower().endswith(valid_extensions):
        continue

    input_path = os.path.join(input_dir, filename)

    # Read image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Could not read {filename}")
        continue

    # Resize image (width, height)
    resized = cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA)

    # Separate filename and extension
    name, ext = os.path.splitext(filename)

    # Remove "_T" from the end of the filename
    if name.endswith("_T"):
        name = name[:-2]

    output_filename = name + ext
    output_path = os.path.join(output_dir, output_filename)

    # Save resized image
    cv2.imwrite(output_path, resized)

    print(f"Saved: {output_filename}")

print("\nAll images processed successfully.")