import os
import cv2
import staintools
import spams
import skimage.io
import histomicstk.preprocessing.color_deconvolution as htk_cd
import histomicstk.preprocessing.color_normalization as htk_cn
import numpy as np
from skimage import exposure
from skimage.io import imread, imsave

#
def vahadane_stain_normalization(input_folder, output_folder, target_image_path):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Vahadane
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target_image)  #


    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"image: {filename}")

            source_image = cv2.imread(img_path)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

            normalized_image = normalizer.transform(source_image)

            output_image_path = os.path.join(output_folder, f"vahadane_{filename}")
            cv2.imwrite(output_image_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))
            print(f"after: {output_image_path}")

def macenko_stain_normalization(input_folder, output_folder, target_image_path):
    """
    Macenko
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Macenko
    normalizer = staintools.StainNormalizer(method='macenko')
    normalizer.fit(target_image)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"image: {filename}")
            source_image = cv2.imread(img_path)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            normalized_image = normalizer.transform(source_image)
            output_image_path = os.path.join(output_folder, f"Macenko_{filename}")
            cv2.imwrite(output_image_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))
            print(f"after image: {output_image_path}")

def ruifrok_stain_normalization(input_folder, output_folder, stain_matrix):
    """
    #Ruifrok
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"image: {filename}")

            source_image = skimage.io.imread(img_path)

            # 3（RGB）
            if source_image.ndim == 2:  # RGB
                source_image = np.stack([source_image] * 3, axis=-1)

            # Ruifrok
            deconvolved = htk_cd.color_deconvolution(source_image, stain_matrix)

            hematoxylin = deconvolved.Stains[:, :, 0]  # Hematoxylin
            eosin = deconvolved.Stains[:, :, 1]  # Eosin

            output_image_path_h = os.path.join(output_folder, f"hematoxylin_{filename}")
            output_image_path_e = os.path.join(output_folder, f"eosin_{filename}")

            skimage.io.imsave(output_image_path_h, hematoxylin)
            skimage.io.imsave(output_image_path_e, eosin)

            print(f"after image: {output_image_path_h}, {output_image_path_e}")

def reinhard_stain_normalization(input_folder, output_folder, target_image_path):
    # Reinhard
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    # Reinhard
    target_mu, target_sigma = staintools.ReinhardNormalizer.get_normalization_stats(target_image)
    normalizer = staintools.ReinhardNormalizer()
    normalizer.set_standardization(target_mu, target_sigma)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"image: {filename}")
            source_image = cv2.imread(img_path)
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            normalized_image = normalizer.transform(source_image)

            # "Reinhard_"
            output_image_path = os.path.join(output_folder, f"Reinhard_{filename}")
            cv2.imwrite(output_image_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))
            print(f"after image: {output_image_path}")

def histogram_matching(input_folder, output_folder, target_image_path):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    target_image = imread(target_image_path)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Pro image: {filename}")

            source_image = imread(img_path)

            if source_image.ndim == 3:
                matched_image = exposure.match_histograms(source_image, target_image, channel_axis=-1)
            else:
                   matched_image = exposure.match_histograms(source_image, target_image)

            # "Matched_"
            output_image_path = os.path.join(output_folder, f"Matched_{filename}")
            imsave(output_image_path, matched_image)
            print(f"after image: {output_image_path}")


def main():
    input_folder = "normalized_inputimages"
    output_folder = "stain_outputimages"
    target_image_path = "target_image.jpg"

    # Macenko
    macenko_stain_normalization(input_folder, output_folder, target_image_path)

    # Vahadane
    vahadane_stain_normalization(input_folder, output_folder, target_image_path)

    stain_matrix = np.array([[0.650, 0.072, 0],  # Hematoxylin
                             [0.704, 0.990, 0],  # Eosin
                             [0.286, 0.105, 0]])

    ruifrok_stain_normalization(input_folder, output_folder, stain_matrix)

   #   Reinhard
   #  # reinhard_stain_normalization(input_folder, output_folder, target_image_path)

    histogram_matching(input_folder, output_folder, target_image_path)

if __name__ == "__main__":
    main()
