import os
import shutil

ad_test_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\hc_split" \
               "/test "
hc_test_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\ad_split" \
               "\\test "

test_save_path = "C:\\Users\\whd\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\test"
def deprefix(root, class_list: list, save_path):
    test_save_path = os.path.join(save_path, 'test')
    os.makedirs(test_save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, 'train')
    os.makedirs(train_save_path, exist_ok=True)

    test_list = []
    train_list = []
    all_list = {'test': test_list, 'train': train_list}
    for class_i in class_list:
        path0 = os.path.join(root, class_i + '_split', 'test')
        test_list.append(path0)

        path1 = os.path.join(root, class_i + '_split', 'train')
        train_list.append(path1)
    for type, list_i in all_list.items():
        for path in list_i:
            for subfolder in os.listdir(path):
                for npy_file in os.listdir(os.path.join(path, subfolder)):
                    shutil.copy(os.path.join(path, subfolder, npy_file), os.path.join(save_path, type))

save_path = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub"
root = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub\\mci_split"
deprefix(root, ['pmci', 'smci'], save_path)