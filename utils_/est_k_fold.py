import os

# 三个折序号文件夹的路径
from utils import load_npy_data
from Dataset import CustomDataset
root = "E:\\datasets\\3dvit\\togather_image_to_sub\\fold\\fold_result"
fold_dirs = ['f1', 'f2', 'f3']
fold_path = [os.path.join(root, f) for f in fold_dirs]
# 类别列表
class_names = ['AD', 'HC']

for i, fold_dir in enumerate(fold_path):
    print(f'Fold {i}:')
    # 训练集文件夹路径
    train_dir = os.path.join(root, fold_dir, 'train')
    # 验证集文件夹路径
    val_dir = os.path.join(root, fold_dir, 'val')

    train_ = CustomDataset(train_dir, split='train')
    val_ = CustomDataset(val_dir, split='val')
    # datanp_train, truenp_train = load_npy_data(train_dir, 'train')
    # datanp_val, truenp_val = load_npy_data(val_dir, 'val')
    # 初始化train和val列表和标签
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    # 遍历AD和HC两个类别文件夹
    for class_name in class_names:
        class_train_dir = os.path.join(train_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        # 遍历训练集文件夹下的npy文件，将路径和标签添加到train_data和train_labels中
        for file_name in os.listdir(class_train_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_train_dir, file_name)
                train_data.append(file_path)
                train_labels.append(class_names.index(class_name))

        # 遍历验证集文件夹下的npy文件，将路径和标签添加到val_data和val_labels中
        for file_name in os.listdir(class_val_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_val_dir, file_name)
                val_data.append(file_path)
                val_labels.append(class_names.index(class_name))

    # 打印train和val列表和标签的长度
    print(f'Train data: {len(train_data)}')
    print(f'Train labels: {len(train_labels)}')
    print(f'Val data: {len(val_data)}')
    print(f'Val labels: {len(val_labels)}\n')
