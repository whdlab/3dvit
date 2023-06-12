import os
import shutil
import random

"""
将某一类的所有被试文件夹分别划分到train或者test文件夹
"""


def split_train_val_in_class(dataset_path, class_list: list, split_rate=0.8):
    for class_i in class_list:

        # 计算每个前缀文件夹中npy文件的数量和总数
        prefix_counts = {}
        total_count = 0
        for file in os.listdir(os.path.join(dataset_path, class_i)):
            for npy_file in os.listdir(os.path.join(dataset_path, class_i, file)):
                if npy_file.endswith(".npy"):
                    prefix = npy_file.split("_")[:3]
                    prefix = "_".join(prefix)
                    # 通过get获取每个前缀键值（个数），若该前缀首次被遍历（键值不存在），则返回0，而数量加1
                    count = prefix_counts.get(prefix, 0)
                    prefix_counts[prefix] = count + 1
                    total_count += 1

        # 初始化训练集和测试集计数器
        train_count = 0
        test_count = 0

        prefixes = list(prefix_counts.keys())
        images_total = sum(prefix_counts.values())
        counts = 0
        while True:
            if counts == images_total:
                break
            file = random.choice(prefixes)
            prefixes.remove(file)
            one_sub_images = os.listdir(os.path.join(dataset_path, class_i, file))
            one_sub_images_nums = len(one_sub_images)
            if train_count < split_rate * total_count:  # train : test = 9:1
                train_path = os.path.join(dataset_path, class_i + '_split', 'train')
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                target_dir = os.path.join(train_path, file)
                train_count += one_sub_images_nums
            else:
                test_count += one_sub_images_nums
                test_path = os.path.join(dataset_path, class_i + '_split', 'test')
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                target_dir = os.path.join(test_path, file)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for image in one_sub_images:
                shutil.copy(os.path.join(dataset_path, class_i, file, image), os.path.join(target_dir, image))
            counts += one_sub_images_nums
        print()
        print("{} Finished! Total npy files: {}. Train set: {}. Test set: {}.".format(class_i, images_total, train_count, test_count))


if __name__ == "__main__":
    dataset_path = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub\\mci_split"
    split_train_val_in_class(dataset_path, ['pmci', 'smci'])
