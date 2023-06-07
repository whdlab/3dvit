#

import shutil
import glob
import os
from utils import mkdir
import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import resize


def images_of_subjects_structure(root_directory, output_directory):
    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)

    # 获取以 wm 开头的 nii 文件路径列表
    nii_files = glob.glob(os.path.join(root_directory, "**/mri/wm*.nii"), recursive=True)

    # 逐个复制 NIfTI 文件到新的文件结构中
    for file_path in nii_files:
        # 从文件路径中提取相关信息，例如主题编号和日期
        subject_id, _, data, S_id, _, _ = file_path.split("\\")[5:]
        # subject_id = file_path.split("\\")[1]
        # data = file_path.split("\\")[3]
        # image_id = (file_path.split("\\")[-1].split('_')[-1]).split('.')[0]

        # 构建目标文件夹路径
        target_folder = os.path.join(output_directory, subject_id, data[0:10])
        os.makedirs(target_folder, exist_ok=True)

        # 构建目标文件路径
        target_file = os.path.join(target_folder, f"wm{subject_id}_{S_id}.nii")

        # 复制文件
        shutil.copyfile(file_path, target_file)

        print("Copied:", target_file)


def all_nii_in_files(root, out):
    os.makedirs(out, exist_ok=True)

    subject_id_list = os.listdir(os.path.join(root, "images"))
    for sub in subject_id_list:
        for i, date in enumerate(os.listdir(os.path.join(root, "images", sub))):
            num = len(os.listdir(os.path.join(root, "images", sub, date)))
            print(num)
            if num != 1:
                print(sub, date)
                with open(os.path.join(out, "than_one_in_one_point.txt"), 'a') as f:
                    f.write(f"{sub}'s {date} point have {num}\n")
            source_image = os.listdir(os.path.join(root, "images", sub, date))[0]
            image = source_image.split('.')[0][:12] + "_" + str(i) + '.nii'
            target_file = os.path.join(out, image)
            shutil.copyfile(os.path.join(root, "images", sub, date, source_image), target_file)
            print("Copied:", target_file)


def modify_pid_column(excel_file, class_):
    """
    在sub ID后面增加表示第几次扫描的后缀，如002_S_0938 -> 002_S_0938_0
    :param csv_file:
    :return:
    """

    df = pd.read_excel(excel_file, sheet_name=class_)
    # df = pd.read_csv(csv_file)

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
    modified_file = excel_file.replace('.xlsx', f'_{class_}_modified.xlsx').split('\\')[-1]
    modified_file = 'E:\\datasets\\ADdata\\originate_datas\\all_new_data\\all_data\\mci\\' + modified_file
    df.to_excel(modified_file, index=False, sheet_name=class_)
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
            # slice_index = 59  # 选择一个切片进行可视化，这里选择第60个切片
            # slice_data1 = img1[:, :, slice_index]
            # slice_data2 = img1[:, 40, :]
            # slice_data = img1[slice_index, :, :]

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


if __name__ == "__main__":
    # step1 往标签信息csv文件的subject id添加_i后缀， 标明第一次的扫描
    # modify_pid_column('E:\\datasets\\ADdata\\originate_datas\\all_new_data\\all_data\\mci\\all_AD&HC.xlsx', class_='pmci')

    # step2 选择每个被试的哪一次扫描作为训练数据
    # filter_pid_suffix('data_file/new_HC_LABEL_190sub_742images.xlsx', suffix='_0',
    # output_filename='./data_file/new_hc_only_fist_subimage.xlsx')

    # step3 重采样，添加标签和mmse，再转化为npy格式u
    # root_image_path = 'E:\\datasets\\ADdata\\originate_datas\\all_new_data\\all_data\\mci'
    # save_path = './datasets/data_npy/112all_mci_npy_data'
    # mmse_label_csv_path = 'C:\\Users\\whd\\PycharmProjects\\3dLenet\data_files\\all_mci_i.xlsx'
    # Generate_Standard_npy_data(root_image_path, save_path, mmse_label_csv_path, class_able_list=['smci', 'pmci'])

    # step4 生成images/sub_id/date/nii_file的结构
    root_directory = "E:\\datasets\\new\\HC\\SMC_in_NEW_NC_images228_sub84"
    output_directory = "E:\\datasets\\new\\HC\\SMC_in_NEW_NC_images228_sub84\\images"
    images_of_subjects_structure(root_directory, output_directory)

    # # step5 将所有的npy文件放到类文件夹下
    root = "E:\\datasets\\new\\HC\\SMC_in_NEW_NC_images228_sub84"
    out = "E:\\datasets\\new\\HC\\SMC_in_NEW_NCdata_images228_sub84"
    all_nii_in_files(root, out)
