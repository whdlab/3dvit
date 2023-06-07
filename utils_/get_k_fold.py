"""
得到k折验证的k个折文件夹
"""
import os
import shutil
import random


# 计算每个前缀文件夹中npy文件的数量和总数

def mkdir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_iamges_sum(file_dataset_path):
    prefix_counts = {}  # 保存各个被试下影像数目的字典
    total_count = 0  # 总数
    for root, dirs, files in os.walk(file_dataset_path):
        for file in files:
            if file.endswith(".npy"):
                prefix = file.split("_")[:3]
                prefix = "_".join(prefix)
                # 通过get获取每个前缀键值（个数），若该前缀首次被遍历（键值不存在），则返回0，而数量加1
                count = prefix_counts.get(prefix, 0)
                prefix_counts[prefix] = count + 1
                total_count += 1
    return prefix_counts, total_count


# 数据集路径
AD_splitedtest_dataset_path = "E:\\datasets\\3dvit\\togather_image_to_sub\\ad_split\\train"
HC_splitedtest_dataset_path = "E:\\datasets\\3dvit\\togather_image_to_sub\\hc_split\\train"
path_list = [AD_splitedtest_dataset_path, HC_splitedtest_dataset_path]

save_fold_path = "E:\\datasets\\3dvit\\togather_image_to_sub\\adfold"
mkdir_path(save_fold_path)

# 三个折序号文件夹的路径
fold_dirs = ['fold_0', 'fold_1', 'fold_2']
# 类别列表
class_names = ['AD', 'HC']

# 构造保存各折影像的文件夹
# for i, fold_dir in enumerate(fold_dirs):
#     print(f'Fold {i}:')
#     # 训练集文件夹路径
#     mkdir_path(os.path.join(save_fold_path, fold_dir))
#     train_dir = os.path.join(save_fold_path, fold_dir, 'train')
#     mkdir_path(train_dir)
#     # 验证集文件夹路径
#     val_dir = os.path.join(save_fold_path, fold_dir, 'val')
#     mkdir_path(val_dir)
#     for cla in class_names:
#         mkdir_path(os.path.join(save_fold_path, fold_dir, cla))
#         mkdir_path(os.path.join(train_dir, cla))
#         mkdir_path(os.path.join(val_dir, cla))

# 初始化训练集和测试集计数器
prefix_counts, total_count = compute_iamges_sum(AD_splitedtest_dataset_path)


fold_list = [os.path.join(save_fold_path, fold) for fold in fold_dirs]
fold_path1, fold_path2, fold_path3 = fold_list

fold1_count = 0
fold2_count = 0
fold3_count = 0
test_count = 0
prefixes = list(prefix_counts.keys())
images_total = sum(prefix_counts.values())
counts = 0
while True:
    if counts == images_total:
        break

    file = random.choice(prefixes)
    prefixes.remove(file)
    one_sub_images = os.listdir(os.path.join(AD_splitedtest_dataset_path, file))
    one_sub_images_nums = len(one_sub_images)

    if fold1_count < (total_count // 3 + total_count % 3):  # train : test = 9:1
        target_dir = os.path.join(fold_path1, file)
        fold1_count += one_sub_images_nums
    else:
        spard_images_two_three_fold = total_count - fold1_count
        if fold2_count < (spard_images_two_three_fold // 2 + total_count % 2):
            target_dir = os.path.join(fold_path2, file)
            fold2_count += one_sub_images_nums
        else:
            fold3_count += one_sub_images_nums
            target_dir = os.path.join(fold_path3, file)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for image in one_sub_images:
        shutil.copy(os.path.join(AD_splitedtest_dataset_path, file, image), os.path.join(target_dir, image))
    counts += one_sub_images_nums

print()
print("Finished! Total npy files: {}. fold1 set: {}. fold2 set: {}. "
      "fold3 set: {}".format(images_total, fold1_count, fold2_count, fold3_count))
