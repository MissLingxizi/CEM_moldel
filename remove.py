
import os
import cv2
import numpy as np


root_folder = "  "

def is_foreground_too_small(image_path, threshold_ratio=0.2):

    image = cv2.imread(image_path)

    if image is None:
        return False
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_pixels = np.count_nonzero(thresh == 0)
    total_pixels = thresh.size

    foreground_ratio = foreground_pixels / total_pixels

    if foreground_ratio < threshold_ratio:
        return True
    return False

folder_deletion_stats = {}

for subdir, dirs, files in os.walk(root_folder):
    deletion_count = 0
    for file in files:
        file_path = os.path.join(subdir, file)

        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if is_foreground_too_small(file_path, threshold_ratio=0.2):
                print(f"Deleting image with small foreground: {file_path}")
                os.remove(file_path)
                deletion_count += 1

    if deletion_count > 0:
        folder_deletion_stats[subdir] = deletion_count
print("\nDeletion statistics for each folder:")
for folder, count in folder_deletion_stats.items():
    print(f"Folder: {folder}, Deleted images: {count}")
print("Finished checking and deleting images.")
