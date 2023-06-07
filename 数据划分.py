import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.2

    data_root = 'E:\\datasets\\data_forlenet\\total_adnc'

    train_root = os.path.join(data_root, "train")
    mk_file(train_root)

    val_root = os.path.join(data_root, "test")
    mk_file(val_root)

    img_npy_path = 'E:\\datasets\\data_forlenet\\total_adnc'
    mk_file(img_npy_path )
    images = os.listdir(img_npy_path)
    num = len(images)
    # 随机采样验证集的索引
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):  # 同时返回迭代对象元素的索引
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            image_path = os.path.join(img_npy_path, image)
            copy(image_path, val_root)
        else:
            # 将分配至训练集中的文件复制到相应目录
            image_path = os.path.join(img_npy_path, image)
            copy(image_path, train_root)
    print("done!")


if __name__ == '__main__':
    main()
