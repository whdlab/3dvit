import random

import nibabel as nib
import os
import numpy as np
import torchvision
from skimage.transform import resize
import pandas as pd
from scipy.ndimage import zoom

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# flip_data = randomflip()
# rotate_180 = randomflip180()
# noise_add = noisy(0.01)
# transform_data = torchvision.transforms.Compose([noise_add, rotate_180, flip_data])

img_path = 'E:\\datasets\\ADdata\\originate_datas\\ad'  # nii文件
save_path = 'E:\\datasets\\3dvit\\128x128x128ad_all_image_per_subj_npy'  # npy文件
mkdir(save_path)
# FLAIR3_000.nii.gz 文件类型命名举例
label_pd = pd.read_excel('E:\\datasets\\data_forlenet\\AD_rename.xlsx',
                         sheet_name='label')  # label excel 存放每一个数据的pid和对应的label，如：pid：100，label：0

for img_name in os.listdir(img_path):
    net_data = []
    pid = img_name.split('.')[0][2:]
    print(pid)
    print(img_name)
    label = label_pd[label_pd['pid'] == str(pid)]['label']
    print(os.path.join(img_path, img_name))
    img_data = nib.load(os.path.join(img_path, img_name))
    img = img_data.get_fdata()
    # 剪除背景
    margin = 10  # 边缘裁剪大小，这里设置为10个体素
    img = img[margin:-margin, margin:-margin, 3:-15]   # (101, 125, 103)
    # 进行重采样
    target_shape = (112, 112, 112)
    zoom_factor = tuple(np.array(target_shape) / np.array(img.shape))
    img = zoom(img, zoom_factor, order=1)
    #     img = nib.load(os.path.join(img_path,img_name).get_fdata() #载入
    img = np.array(img).astype(np.float32)
    # nomalization
    if np.min(img) < np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    if np.unique(label == 1):
        label_data = 1
        net_data.append([img, label_data])
        np.save(os.path.join(save_path, pid), net_data)  # 保存
    if np.unique(label == 0):
        label_data = 0
        net_data.append([img, label_data])
        np.save(os.path.join(save_path, pid), net_data)  # 保存
print('Done!')
