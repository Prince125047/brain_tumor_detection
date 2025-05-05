import os
import cv2
import albumentations as A
from tqdm import tqdm

# Path to processed Training folder
TRAINING_PATH = 'preprocess_dataset/Training'

# Class names (folders inside Training)
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Albumentations augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])

# Function to augment images
def augment_images():
    for cls in classes:
        class_folder = os.path.join(TRAINING_PATH, cls)

        for img_file in tqdm(os.listdir(class_folder), desc=f'Augmenting {cls}'):
            img_path = os.path.join(class_folder, img_file)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Apply augmentations
                augmented = augment(image=img)['image']

                # Save augmented image
                new_filename = 'aug_' + img_file
                save_path = os.path.join(class_folder, new_filename)
                cv2.imwrite(save_path, augmented)

            except Exception as e:
                print(f"Failed augmenting {img_path}: {e}")

if __name__ == '__main__':
    augment_images()
