import os
import shutil

ad_test_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\hc_split/test"
hc_test_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\ad_split\\test"

test_save_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\test"
os.makedirs(test_save_path, exist_ok=True)

path_list = [ad_test_path, hc_test_path]
for path in path_list:
    for subfolder in os.listdir(path):
        for npy_file in os.listdir(os.path.join(path, subfolder)):
            shutil.copy(os.path.join(path, subfolder, npy_file), test_save_path)