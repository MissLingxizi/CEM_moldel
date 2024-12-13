

import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path):

    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
    return images

def apply_horizontal_flip(image):

    return cv2.flip(image, 1)

def apply_vertical_flip(image):

    return cv2.flip(image, 0)

def apply1_random_rotation(image):

    angle = random.randint(-90, 90)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def apply_random_zoom(image):

    zoom_factor = random.uniform(0.8, 1.2)
    rows, cols = image.shape[:2]
    zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(zoomed_image, (cols, rows), interpolation=cv2.INTER_LINEAR)

def apply_random_crop(image):

    scale = random.uniform(0.8, 1.0)
    rows, cols = image.shape[:2]
    new_height, new_width = int(scale * rows), int(scale * cols)
    x = random.randint(0, cols - new_width)
    y = random.randint(0, rows - new_height)
    return image[y:y + new_height, x:x + new_width]

def apply_translation(image):

    rows, cols = image.shape[:2]
    x_translation = random.randint(-50, 50)
    y_translation = random.randint(-50, 50)
    M = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    return cv2.warpAffine(image, M, (cols, rows))

def apply1_random_brightness(image):

    brightness_factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply1_random_contrast(image):

    contrast_factor = random.uniform(0.5, 1.5)
    mean = np.mean(image)
    return np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

def apply1_random_hue(image):

    hue_shift = random.randint(-20, 20)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_gaussian_noise(image):

    noise = np.random.normal(loc=0, scale=25, size=image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def apply_color_space_transform(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply1_combined_transform(image):

    transformed_image = apply1_random_rotation(image)
    transformed_image = apply1_random_brightness(transformed_image)
    transformed_image = apply1_random_contrast(transformed_image)
    return transformed_image

def normalize_image(image):

    return image.astype(np.float32) / 255.0
def apply_augmentation_to_image(image, augmentation_functions, num_functions):

    augmented_images = []
    method_prefixes = []

    chosen_functions = random.sample(augmentation_functions, num_functions)

    for func in chosen_functions:
        augmented_img = func(image.copy())
        method_prefixes.append(func.__name__)
        augmented_images.append(augmented_img)

    return augmented_images, method_prefixes

def display_random_augmented_images(original_image, augmented_images, prefixes):

    num_images = len(augmented_images)
    plt.figure(figsize=(12, 6))


    plt.subplot(1, num_images + 1, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')


    for i in range(num_images):
        plt.subplot(1, num_images + 1, i + 2)
        plt.imshow(cv2.cvtColor(augmented_images[i], cv2.COLOR_BGR2RGB))
        plt.title(prefixes[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def normalize_images(images, output_folder):

    normalized_images = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image in enumerate(images):
        normalized_image = image.astype(float) / 255.0
        normalized_images.append(normalized_image)

        filename = f'normalized_{i + 1}.jpg'

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))

    return normalized_images

def display_image_comparison(original_image, normalized_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor((normalized_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Normalized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    input_folder = 'D:\\data1'
    output_folder = 'D:\\data2'
    num_augmentations = 2
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = load_images_from_folder(input_folder)

    augmentation_functions = [
        apply_horizontal_flip,
        apply_vertical_flip,
        apply1_random_rotation,
        apply_random_zoom,
        apply_random_crop,
        apply_translation,
        apply1_random_brightness,
        apply1_random_contrast,
        apply1_random_hue,
        apply_gaussian_noise,
        apply_color_space_transform,
        apply1_combined_transform
    ]

    for idx, image in enumerate(images):

        augmented_images, prefixes = apply_augmentation_to_image(image, augmentation_functions, num_augmentations)

        for i, augmented_img in enumerate(augmented_images):
            prefix = prefixes[i]
            filename = f'{prefix}_{idx + 1}.jpg'
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, augmented_img)


        if idx == random.randint(0, len(images) - 1):
            display_random_augmented_images(image, augmented_images, prefixes)

    print(f'finish save "{output_folder}" ')

