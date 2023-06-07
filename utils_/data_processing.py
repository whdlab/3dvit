import os

import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import resize
# 读取CSV文件
# df = pd.read_csv('C:\\Users\\whd\\Desktop\\css.csv')
#
# # 获取A和B列的唯一值
# missing_a_values = df[df['A'].isnull() & df['B'].notnull()]['B'].values
#
# # 将数据写入txt文件
# with open('./missing_a_values.txt', 'w') as file:
#     for value in missing_a_values:
#         file.write(str(value) + '\n')
from utils import mkdir


def modify_pid_column(csv_file):
    """
    在sub ID后面增加表示第几次扫描的后缀，如002_S_0938 -> 002_S_0938_0
    :param csv_file:
    :return:
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 获取'pid'列中的重复数据
    duplicates = df[df['pid'].duplicated(keep=False)]
    # 根据只出现一次的数据添加'_0'
    value_counts = df['pid'].value_counts()
    single_occurrences = value_counts[value_counts == 1].index

    for value in single_occurrences:
        df.loc[df['pid'] == value, 'pid'] = f"{value}_0"

    # 根据重复次数进行重命名
    counter = {}
    for index, row in duplicates.iterrows():
        pid = row['pid']
        if pid not in counter:
            counter[pid] = 0
        else:
            counter[pid] += 1
        new_pid = f"{pid}_{counter[pid]}"
        df.at[index, 'pid'] = new_pid
    # 将修改后的数据保存为CSV文件
    modified_file = csv_file.replace('.csv', '_modified.csv').split('\\')[-1]
    modified_file = './data_file/' + modified_file
    df.to_csv(modified_file, index=False)
    print(f"Modified file '{modified_file}' created.")


def filter_pid_suffix(filename, suffix='_0', output_filename='ad_only_fist_subimage.xlsx'):
    """
    在总的xlsx文件中选择第几次扫描作为输入数据
    :param filename: 存有标签信息的xlsx文件路径
    :param suffix: 后缀，需要获取的被试的第几次扫描, 例如002_S_0938_0,002_S_0938_1,002_S_0938_2...中_0, _1...
    :param output_filename: 输出的xlsx
    :return:
    """
    # 读取.xlsx文件
    df = pd.read_excel(filename)

    # 根据条件筛选行
    filtered_rows = df[df['pid'].astype(str).str.endswith(suffix)]

    # 创建一个新的DataFrame来保存筛选后的行数据
    new_df = pd.DataFrame(filtered_rows)

    # 将数据保存为新的.xlsx文件
    new_df.to_excel(output_filename, index=False)

    print(f"Filtered rows saved in '{output_filename}'.")

def Generate_Standard_npy_data(Nii_file_root_path, save_npy_root_path,
                               mmse_label_csv_path, class_able_list=None):
    """

    :param Nii_file_root_path:
    :param save_npy_root_path:
    :param mmse_label_csv_path: 包含label和mmse分数的csv
    :param class_able_list:
    :return:
    """
    if class_able_list is None:
        class_able_list = ['ad', 'hc', 'smci', 'pmci']
    for class_i in class_able_list:
        class_file_path = os.path.join(Nii_file_root_path, class_i)
        class_save_path = os.path.join(save_npy_root_path, class_i)

        mkdir(class_save_path)
        all_data_list = os.listdir(os.path.join(class_file_path))
        label_pd = pd.read_excel(mmse_label_csv_path, sheet_name=class_i)
        targets_pid = label_pd['pid']
        for pid_i in targets_pid:
            net_data = []
            img_name = 'wm' + pid_i + '.nii'
            if img_name not in all_data_list:
                with open(os.path.join(Nii_file_root_path, f"{class_i}_nii_File_no_exist.txt"), 'a') as f:
                    f.write(img_name)
                continue
            # label excel 存放每一个数据的pid和对应的label，如：pid：002_S_0938，label：0, MMSE Total Score:26
            print(pid_i)
            print(img_name)

            label = label_pd[label_pd['pid'] == str(pid_i)]['label']
            mmse = label_pd[label_pd['pid'] == str(pid_i)]['MMSE Total Score']

            print(os.path.join(class_file_path, img_name))
            img_data = nib.load(os.path.join(class_file_path, img_name))
            img = img_data.get_fdata()
            # 剪除背景
            margin = 10  # 边缘裁剪大小，这里设置为10个体素
            img = img[margin:-margin, margin:-margin, 3:-15]  # (101, 125, 103)
            # 进行重采样
            target_shape = (112, 112, 112)
            zoom_factor = tuple(np.array(target_shape) / np.array(img.shape))
            img1 = zoom(img, zoom_factor, order=1)
            # 精度调整
            img = np.array(img1).astype(np.float32)

            # 可视化裁剪后的影像
            slice_index = 59  # 选择一个切片进行可视化，这里选择第60个切片
            slice_data1 = img1[:, :, slice_index]
            slice_data2 = img1[:, 40, :]
            slice_data = img1[slice_index, :, :]

            # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            #
            # axs[0][0].imshow(slice_data, cmap='gray')
            # axs[0][0].set_title('sagittal plane')
            # axs[0][1].imshow(slice_data1, cmap='gray')
            # axs[0][1].set_title('Cross-sectional')
            # axs[1][0].imshow(slice_data2, cmap='gray')
            # axs[1][0].set_title('coronal plane')
            # plt.axis('off')
            # plt.show()
            # nomalization
            if np.min(img) < np.max(img):
                img = img - np.min(img)
                img = img / np.max(img)
            if np.unique(label == 1):
                label_data = 1
                net_data.append([img, label_data, mmse])
                np.save(os.path.join(class_save_path, pid_i), net_data)
            if np.unique(label == 0):
                label_data = 0
                net_data.append([img, label_data, mmse])
                np.save(os.path.join(class_save_path, pid_i), net_data)
        print('Done!')


# 调用函数
# filter_pid_suffix('data_file/new_HC_LABEL_190sub_742images.xlsx', suffix='_0', output_filename='./data_file/new_hc_only_fist_subimage.xlsx')
# modify_pid_column('C:\\Users\\whd\\Desktop\\ad_all.csv')
root_image_path = 'E:\\datasets\\ADdata\\originate_datas\\all_new_data\\all_data'
save_path = 'data_npy/112all_ad&hc_npy_data'
mmse_label_csv_path = 'data_file/AD295&HC346_LABEL_mmse_753_1344.xlsx'
Generate_Standard_npy_data(root_image_path, save_path, mmse_label_csv_path, class_able_list=['ad'])