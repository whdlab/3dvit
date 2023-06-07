"""
--------stage1---------
function:
1、将nii文件数据加载成数组， 根据AD_rename.xlsx包含的sub_id和标签在数据数组内嵌入标签数据，转化为npy格式
2、放缩成想要的大小

"""
import nibabel as nib
import os
import numpy as np
import torchvision
from skimage.transform import resize
import pandas as pd


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    img = resize(img, (128, 128, 128), order=0)  # 将图像大小进行统一缩放，方便输入网络，分别为（h,w,c）,可根据自己的数据集来更改

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
