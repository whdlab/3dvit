import os
import shutil
import glob
from random import sample

def Downsample(root_folder, target_remain_nums, target_remove_later_i, target_root):
    total_nums = 0
    per_sub_num_list = []
    all_remove_images = []
    for i in os.listdir(root_folder):
        per_sub_num = len(os.listdir(os.path.join(root_folder, i)))
        total_nums += per_sub_num
        per_sub_num_list.append(per_sub_num)
    max_num = max(per_sub_num_list)
    os.makedirs(target_root, exist_ok=True)

    target_remove_nums = total_nums - target_remain_nums
    for j in range(max_num - target_remove_later_i):
        removed_images = glob.glob(os.path.join(root_folder, f"**/*_{target_remove_later_i + j}.npy"), recursive=True)
        num = len(all_remove_images)
        if len(removed_images) >= target_remove_nums - num:
            removed_images = sample(removed_images, target_remove_nums - num)
        all_remove_images.extend(removed_images)

        num_list_remove = ["_" + str(k + target_remove_later_i) for k in range(j + 1)]
        if target_remove_nums <= num:
            break
    print("舍去后缀为：", num_list_remove)
    print("total nums:", total_nums)
    print("total remove:", num)
    print("remain: ", total_nums - num)
    for i in all_remove_images:
        shutil.move(i, target_root)
    print(f"moved:{all_remove_images}")


if __name__ == "__main__":
    root_folder = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub\\smci"
    target_folder = 'E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub\\smci_remove_target'
    Downsample(root_folder, 1407, 3, target_folder)
