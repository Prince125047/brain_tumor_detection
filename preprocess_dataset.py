import os
import cv2
from tqdm import tqdm
from extractor import extract_brain_contour

# Paths
SOURCE_PATH = 'dataset'  # Example: ./dataset
DESTINATION_PATH = 'preprocess_dataset'  # Example: ./final_dataset

# Class folders
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
folders = ['Training', 'Testing']

def preprocess_images():
    for folder in folders:
        for cls in classes:
            source_folder = os.path.join(SOURCE_PATH, folder, cls)
            destination_folder = os.path.join(DESTINATION_PATH, folder, cls)

            os.makedirs(destination_folder, exist_ok=True)

            for img_file in tqdm(os.listdir(source_folder), desc=f'Processing {folder}/{cls}'):
                img_path = os.path.join(source_folder, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    # Apply contour cropping
                    cropped_img = extract_brain_contour(img)

                    # Resize to 256x256
                    resized_img = cv2.resize(cropped_img, (256, 256))

                    # Save
                    save_path = os.path.join(destination_folder, img_file)
                    cv2.imwrite(save_path, resized_img)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == '__main__':
    preprocess_images()
