import os
import random
import shutil

import numpy as np
import glob

root_folder = "C:\\Users\\whd\\PycharmProjects\\3dLenet\\utils_\\datasets\\data_npy\\112all_mci_npy_data\\togather_image_to_sub\\smci"
target_num_images = 1042

target_folder = 'C:\\Users\\whd\\PycharmProjects\\3dLenet\\utils_\\datasets\\data_npy\\112all_mci_npy_data\\togather_image_to_sub\\pmci_remove_target'
os.makedirs(target_folder, exist_ok=True)

remaining_images = []
removed_images1 = glob.glob(os.path.join(root_folder, "**/*_3.npy"), recursive=True)
removed_images2 = glob.glob(os.path.join(root_folder, "**/*_5.npy"), recursive=True)
removed_images3 = glob.glob(os.path.join(root_folder, "**/*_10.npy"), recursive=True)
removed_images = removed_images1 + removed_images2 + removed_images3
removed_images_nums = len(removed_images)
# 遍历根文件夹下的子文件夹
for i in removed_images:
    shutil.move(i, target_folder)
print(f"moved:{removed_images_nums}")