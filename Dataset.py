import os
import random

# import torchio
# from torchio import *
# import volumentations
from skimage import transform, exposure
import numpy as np
import skimage
import torch
from skimage import exposure
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import nibabel as nib
# from volumentations import *
import torch
from torch.utils.data import Dataset, DataLoader
from data_add import transform_data
from data_aug import *


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transform=None):
        self.data = data_root
        self.label = data_label
        self.transform = transform

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]

        if self.transform is not None:
            data = self.transform(data)

        return torch.tensor(data).float(), torch.tensor(labels).float()

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


class CustomDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        self.data_files = os.listdir(self.data_dir)
        self.transform = transforms.Compose([
            xyz_rotate(-10, 10, rate=0.5),
            flip(rate=0.5),
            # mask(rate=0.5, mask_nums=2, intersect=False),
            contrast(),
        ])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.data_files[idx]), allow_pickle=True)
        image = data[0][0]
        label = data[0][1]

        # if self.split == 'train':
        #     # 数据增强处理
        #     image = self.transform(image)
        return torch.tensor(image).float().unsqueeze(0), torch.tensor(label).float()


#
# if __name__ == '__main__':
#     # 读取NII文件
#     nii_file = nib.load(
#         'C:\\Users\\whd\\PycharmProjects\\DA-MIDL-main\\resample\\ad_last_image_per_subj\\wm002_S_0938_2.nii')
#     # 获取sMRI数据
#     smri_data = nii_file.get_fdata()
#     nii_file1 = nib.load(
#         'C:\\Users\\whd\\PycharmProjects\\DA-MIDL-main\\resample\\ad_last_image_per_subj\\wm002_S_0938_2.nii')
#     # 获取sMRI数据
#     smri_data1 = nii_file1.get_fdata()
#     # 获取sMRI数据的形状
#     smri_shape = smri_data.shape
#
#
# # 定义数据扩增类
# class randomflip180(object):
#     def __call__(self, data):
#         if random.uniform(0, 1) > 0.5:
#             data = np.flip(data, axis=0)
#             data = np.flip(data, axis=1)
#         if random.uniform(0, 1) > 0.5:
#             data = np.flip(data, axis=1)
#             data = np.flip(data, axis=2)
#         if random.uniform(0, 1) > 0.5:
#             data = np.flip(data, axis=2)
#             data = np.flip(data, axis=0)
#         return data
#
#
# class noisy(object):
#     def __init__(self, radio):
#         self.radio = radio
#
#     def __call__(self, data):
#         l, w, h = data.shape
#         num = int(l * w * h * self.radio)
#         for _ in range(num):
#             x = np.random.randint(0, l)
#             y = np.random.randint(0, w)
#             z = np.random.randint(0, h)
#             data[x, y, z] = data[x, y, z] + np.random.uniform(0, self.radio)
#         return data
#
#
# class RandomCrop3D(object):
#     def __init__(self, output_size, padding=False):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size, output_size)
#         else:
#             assert len(output_size) == 3
#             self.output_size = output_size
#         self.padding = padding
#
#     def __call__(self, data):
#         x, y, z = data.shape
#         new_x, new_y, new_z = self.output_size
#
#         if x == new_x and y == new_y and z == new_z:
#             return data
#
#         x_margin = x - new_x
#         y_margin = y - new_y
#         z_margin = z - new_z
#
#         x_min = random.randint(0, x_margin) if x_margin > 0 else 0
#         y_min = random.randint(0, y_margin) if y_margin > 0 else 0
#         z_min = random.randint(0, z_margin) if z_margin > 0 else 0
#
#         x_max = x_min + new_x
#         y_max = y_min + new_y
#         z_max = z_min + new_z
#
#         cropped = data[x_min:x_max, y_min:y_max, z_min:z_max]
#
#         if self.padding:
#             padding_x = (0, 0)
#             padding_y = (0, 0)
#             padding_z = (0, 0)
#
#             if x_margin < 0:
#                 padding_x = (-x_margin // 2, (-x_margin + 1) // 2)
#             if y_margin < 0:
#                 padding_y = (-y_margin // 2, (-y_margin + 1) // 2)
#             if z_margin < 0:
#                 padding_z = (-z_margin // 2, (-z_margin + 1) // 2)
#
#             padded = np.pad(cropped, ((0, 0), padding_x, padding_y, padding_z), mode='constant', constant_values=0)
#             return padded
#         else:
#             return cropped
#
#
# def random_crop_3d(img, crop_size):
#     """
#     从 3D 图像中随机裁剪一个子体积
#
#     参数：
#         img (ndarray): 3D 图像
#         crop_size (tuple): 裁剪的子体积大小
#
#     返回：
#         ndarray: 裁剪后的 3D 图像子体积
#     """
#     assert img.ndim == 3, "输入图像必须是 3D"
#     assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1] and crop_size[2] <= img.shape[2], "裁剪尺寸大于图像尺寸"
#     x_range = img.shape[0] - crop_size[0]
#     y_range = img.shape[1] - crop_size[1]
#     z_range = img.shape[2] - crop_size[2]
#     x_start = random.randint(0, x_range)
#     y_start = random.randint(0, y_range)
#     z_start = random.randint(0, z_range)
#     cropped_img = img[x_start:x_start + crop_size[0], y_start:y_start + crop_size[1], z_start:z_start + crop_size[2]]
#     return cropped_img
#
#
#
# # 对sMRI数据进行扩增
# transform_byme = transforms.Compose([
#     randomflip180(),
#     # 其他的数据扩增方法
# ])


# smri_data_path = "C:\\Users\\whd\\Desktop\\MPSFFA-main\\data_npy\\112all_ad&hc_npy_data\\togather_image_to_sub\\train\\002_S_0295_0.npy"
# smri_data = np.load(smri_data_path, allow_pickle=True)[0][0]
# smri_shape = smri_data.shape
#
# tran1 = xy_rotate(0, 60, rate=0.5)
# tran2 = xyz_rotate(-10, 10, rate=0.1)
# tran3 = gamma_adjust(0.5, 0.7, rate=0.1)
# tran4 = contrast()
# tran5 = sample()
# tran6 = mask(10, intersect=False)
# tran7 = equa_hist()
# tran8 = flip(rate=0.1)
#
# transform_byme = transforms.Compose([
#     xyz_rotate(-10, 10, rate=0.5),
#     flip(rate=0.5),
#     mask(rate=0.5, mask_nums=2, intersect=False),
#     contrast(),
# ])
# data_sug1 = tran1(smri_data)
# data_sug2 = tran2(smri_data)
# data_sug3 = tran3(smri_data)
# data_sug4 = tran4(smri_data)
# data_sug5 = tran5(smri_data)
# data_sug5_shape = data_sug5.shape
# data_sug6 = tran6(smri_data)
# data_sug7 = tran7(smri_data)
# data_sug8 = tran8(smri_data)
#
# data_sugji = transform_byme(smri_data)
# # data_sug2 = skimage.transform.rotate(smri_data, 60)  # 旋转60度，不改变大小
# # data_sug3 = exposure.exposure.adjust_gamma(smri_data, gamma=0.7)  # 变亮
# # data_sug4 = transform_byme(smri_data)
#
# # 可视化sMRI数据的扩增前后
# fig1, axs1 = plt.subplots(figsize=(10, 10))
# axs1.imshow(data_sugji[:, :, smri_shape[2] // 2], cmap='gray')
# axs1.set_title('compose')
#
# fig, axs = plt.subplots(3, 4, figsize=(20, 20))
# axs[0][3].imshow(smri_data[:, :, smri_shape[2] // 2], cmap='gray')
# axs[0][3].set_title('Original')
# axs[0][0].imshow(data_sug1[:, :, smri_shape[2] // 2], cmap='gray')
# axs[0][0].set_title('rotate_xy')
# axs[0][1].imshow(data_sug2[:, :, smri_shape[2] // 2], cmap='gray')
# axs[0][1].set_title('rotate_xyz')
# axs[0][2].imshow(data_sug3[:, :, smri_shape[2] // 2], cmap='gray')
# axs[0][2].set_title('adjust_gamma')
# axs[1][0].imshow(data_sug4[:, :, smri_shape[2] // 2], cmap='gray')
# axs[1][0].set_title('contrast')
# axs[1][1].imshow(data_sug5[:, :, data_sug5_shape[2] // 2], cmap='gray')
# axs[1][1].set_title('sample')
# axs[1][2].imshow(data_sug6[:, :, smri_shape[2] // 2], cmap='gray')
# axs[1][2].set_title('mask')
# axs[1][3].imshow(data_sug7[:, :, smri_shape[2] // 2], cmap='gray')
# axs[1][3].set_title('equa_hist')
# axs[2][0].imshow(data_sug8[:, :, smri_shape[2] // 2], cmap='gray')
# axs[2][0].set_title('flip')
# plt.show()
