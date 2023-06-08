import os
import shutil

ad_test_path = "C:\\Users\\whd\\PycharmProjects\\3dLenet\\utils_\\datasets\\data_npy\\112all_mci_npy_data\\togather_image_to_sub\\smci_split\\train"
hc_test_path = "C:\\Users\\whd\\PycharmProjects\\3dLenet\\utils_\\datasets\\data_npy\\112all_mci_npy_data\\togather_image_to_sub\\pmci_split\\train"

test_save_path = "C:\\Users\\whd\\PycharmProjects\\3dLenet\\utils_\\datasets\\data_npy\\112all_mci_npy_data\\togather_image_to_sub\\train"
os.makedirs(test_save_path, exist_ok=True)

path_list = [ad_test_path, hc_test_path]
for path in path_list:
    for subfolder in os.listdir(path):
        for npy_file in os.listdir(os.path.join(path, subfolder)):
            shutil.copy(os.path.join(path, subfolder, npy_file), test_save_path)