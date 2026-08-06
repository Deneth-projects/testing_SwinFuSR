import cv2
import os

def process_lr_folder_os():
    input_dir = "/LR_ground_truth"
    output_dir = "/LR"
    
    # Ensure output folder exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Combine folder path and filename
            full_input_path = os.path.join(input_dir, filename)
            
            img = cv2.imread(full_input_path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                print("This image Not found : ",full_input_path)
                continue

            #cv2.resize expects (width, height)
            img_scaled = cv2.resize(img, (60, 80), interpolation=cv2.INTER_AREA)

            full_output_path = os.path.join(output_dir, filename)
            cv2.imwrite(full_output_path, img_scaled)
            print(f"Processed: {filename}")

# Run the function
process_lr_folder_os()
