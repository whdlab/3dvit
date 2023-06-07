import nibabel as nib
import os
import numpy as np
import torchvision
from skimage.transform import resize
import pandas as pd


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


img_path = 'E:\\datasets\\BMCP\\p-sorce\\ADNI\\MRI_278data\\PostProcessed\\AD'  # nii文件
save_path = 'C:\\Users\\whd\\Desktop\\adddata'  # npy文件
mkdir(save_path)
for img_name in os.listdir(img_path):
    net_data = []
    name = img_name[8:19] + "0"
    print(name)
    print(img_name)
    print(os.path.join(img_path, img_name))
    img_data = nib.load(os.path.join(img_path, img_name))
    img = img_data.get_fdata()
    img = resize(img, (128, 128, 128), order=0)  # 将图像大小进行统一缩放，方便输入网络，分别为（h,w,c）,可根据自己的数据集来更改

    #     img = nib.load(os.path.join(img_path,img_name).get_fdata() #载入
    img = np.array(img).astype(np.float32)
    # nomalization
    if np.min(img) < np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)

    label_data = 0
    net_data.append([img, label_data])
    np.save(os.path.join(save_path, name), net_data)  # 保存
print('Done!')
