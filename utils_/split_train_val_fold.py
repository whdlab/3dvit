import os
import shutil

root = "E:\\datasets\\3dvit\\togather_image_to_sub\\fold"
# 设置文件夹名称
folds = ['fold_f1', 'fold_f2', 'fold_f3']

def create_train_val_folders(folds, root="E:\\datasets\\3dvit\\togather_image_to_sub\\fold",
                             train_dir='train', val_dir='val'):
    """
    将folds列表中的折文件夹两两合并作为训练集，另一个作为验证集，并将它们存储在f1，f2，f3三个文件夹下，
    即f1，f2，f3文件夹下包含train和val文件夹。

    参数：
    folds: 保存各折文件夹名称的列表。
    train_dir: 训练集文件夹名称，默认为'train'。
    val_dir: 验证集文件夹名称，默认为'val'。
    """
    for i in range(len(folds)):
        # 获取当前折文件夹的路径
        fold_path = os.path.join(root, folds[i])

        # 创建训练集和验证集文件夹
        train_path = os.path.join(root, 'f' + str(i + 1), train_dir)
        os.makedirs(train_path, exist_ok=True)

        val_path = os.path.join(root, 'f' + str(i + 1), val_dir)
        os.makedirs(val_path, exist_ok=True)

        # 将另外两个折文件夹的内容复制到训练集文件夹中
        for j in range(len(folds)):
            if j != i:
                other_path = os.path.join(root, folds[j])
                for subfolder in os.listdir(other_path):
                    src = os.path.join(other_path, subfolder)
                    dst = os.path.join(train_path, subfolder)
                    shutil.copytree(src, dst)

        # 将当前折文件夹的内容复制到验证集文件夹中
        for subfolder in os.listdir(fold_path):
            src = os.path.join(fold_path, subfolder)
            dst = os.path.join(val_path, subfolder)
            shutil.copytree(src, dst)


def split_train_val(fold_dir, train_dir='train', val_dir='val',
                    root="E:\\datasets\\3dvit\\togather_image_to_sub\\fold"):
    # 获取所有折文件夹的路径
    fold_paths = [os.path.join(root, fold_name) for fold_name in fold_dir]

    for i in range(len(fold_paths)):
        # 取出第i个折文件夹作为验证集
        val_fold_path = fold_paths[i]
        val_prefix_paths = [os.path.join(val_fold_path, prefix_name) for prefix_name in os.listdir(val_fold_path)]
        val_npy_files = []
        for prefix_path in val_prefix_paths:
            val_npy_files += [os.path.join(prefix_path, npy_file) for npy_file in os.listdir(prefix_path) if npy_file.endswith('.npy')]

        # 将其余折文件夹的所有前缀文件夹中的npy文件作为训练集
        train_prefix_paths = []
        train_npy_files = []
        for j in range(len(fold_paths)):
            if j != i:
                train_fold_path = fold_paths[j]
                train_prefix_paths += [os.path.join(train_fold_path, prefix_name) for prefix_name in os.listdir(train_fold_path)]
        for prefix_path in train_prefix_paths:
            train_npy_files += [os.path.join(prefix_path, npy_file) for npy_file in os.listdir(prefix_path) if npy_file.endswith('.npy')]

        # 将训练集和验证集分别保存到train_dir和val_dir下
        train_path = os.path.join(root, "fold_result", 'f' + str(i + 1), train_dir)
        os.makedirs(train_path, exist_ok=True)

        val_path = os.path.join(root, "fold_result", 'f' + str(i + 1), val_dir)
        os.makedirs(val_path, exist_ok=True)

        for npy_file in train_npy_files:
            shutil.copy(npy_file, train_path)
        for npy_file in val_npy_files:
            shutil.copy(npy_file, val_path)


if __name__ == "__main__":
    # create_train_val_folders(folds)
    split_train_val(folds)