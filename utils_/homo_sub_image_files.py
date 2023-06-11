import os
import shutil

"""
------stage2---------
Function：将ad和nc类别的所有npy文件中属于同一被试的所有时间点扫描分到以被试id为前缀的文件夹中
**确保例E:/datasets/3dvit/chongfu下有包含ad和nc两个文件夹，分别包含两个类别所有被试的所有影像的npy格式（float32)
output_dir 为保存各个被试前缀文件夹的类别文件夹
"""


def homo_sub_image_files(input_dir_root, class_list, output_dir_root):
    for class_i in class_list:
        output_class_i_path = os.path.join(output_dir_root, class_i)
        input_class_i_path = os.path.join(input_dir_root, class_i)
        if not os.path.exists(output_class_i_path):
            os.makedirs(output_class_i_path)
        # 遍历输入文件夹中的所有npy文件
        for filename in os.listdir(input_class_i_path):
            if filename.endswith(".npy"):
                # 提取前缀
                prefix = filename.split("_")[:3]
                prefix = "_".join(prefix)

                # 创建对应的输出文件夹（如果不存在的话）
                prefix_dir = os.path.join(output_class_i_path, prefix)
                if not os.path.exists(prefix_dir):
                    os.makedirs(prefix_dir)

                # 将文件移动到对应的输出文件夹中
                src_path = os.path.join(input_class_i_path, filename)
                dst_path = os.path.join(prefix_dir, filename)
                shutil.copy(src_path, dst_path)
    print("done")


if __name__ == '__main__':
    input_root = "E:\\datasets\\using_datasets\\112all_mci_npy_data"
    class_list = ['smci', 'pmci']
    out_root = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub"
    homo_sub_image_files(input_root, class_list, out_root)