import os
from PIL import Image
import judge_bank

def split_image_into_patches(image_path, output_folder, patch_size=(380, 380)):

    os.makedirs(output_folder, exist_ok=True)


    with Image.open(image_path) as img:
   
        width, height = img.size


        num_patches_width = (width - patch_size[0]) // patch_size[0]
        num_patches_height = (height - patch_size[1]) // patch_size[1]
        save_name = image_path.split('\\')[-1].split('.')[0]

        for i in range(num_patches_width):
            for j in range(num_patches_height):
         
                left = i * patch_size[0]
                top = j * patch_size[1]
              
                patch = img.crop((left, top, left + patch_size[0], top + patch_size[1]))
 
                if judge_bank.get_blank_rate(patch) < 1:
                    patch_path = os.path.join(output_folder, f"{save_name}_{i}_{j}.jpg")
                    try:
                        patch.save(patch_path)
                    except OSError:
                        print(999)



def process_folder(input_folder, output_folder, patch_size=(380, 380)):

    for filename in os.listdir(input_folder):
  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

            image_path = os.path.join(input_folder, filename)

            split_image_into_patches(image_path, output_folder, patch_size=patch_size)


for dir in os.listdir('_source'):
    os.makedirs('_patch_'+dir)
    input_folder1 = "xx"+dir  # 
    output_folder1 = "xxx"+dir  # 
    process_folder(input_folder1, output_folder1)
