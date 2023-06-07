
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nibabel.viewers import OrthoSlicer3D

# 加载NIfTI格式的sMRI影像
from skimage.transform import resize

image_path = 'E:\\datasets\\ADdata\\originate_datas\\all_new_data\\all_data\\ad\\wm002_S_0619_0.nii'  # 将"path_to_image.nii"替换为实际的影像路径
image = nib.load(image_path)
data = image.get_fdata()

# 获取裁剪后影像的数据和空间信息

affine = image.affine
# 裁剪空白背景
margin = 10  # 边缘裁剪大小，这里设置为10个体素
cropped_data = data[margin:-margin, margin:-margin, 3:-15]
# 创建裁剪后的NIfTI影像对象
cropped_image = nib.Nifti1Image(cropped_data, image.affine, image.header)
cropped_shape = cropped_image.shape
print("裁剪后的影像尺寸：", cropped_shape)
# 保存裁剪后的影像为NIfTI格式
output_path = './path_to_output.nii'  # 将"path_to_output.nii"替换为输出影像的路径
nib.save(cropped_image, output_path)

# 可视化裁剪后的影像
slice_index = 59  # 选择一个切片进行可视化，这里选择第60个切片
slice_data1 = cropped_data[:, :, slice_index]
slice_data2 = cropped_data[:, slice_index, :]
slice_data = cropped_data[slice_index, :, :]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0][0].imshow(slice_data, cmap='gray')
axs[0][0].set_title('randomflip180')
axs[0][1].imshow(slice_data1, cmap='gray')
axs[0][1].set_title('rotate60')
axs[1][0].imshow(slice_data2, cmap='gray')
axs[1][0].set_title('rotate61')


#
# 将影像进行重采样到128x128x128大小
# 目标重采样尺寸
target_shape = (112, 112, 112)
img = resize(cropped_data, target_shape, order=0)
# 计算重采样比例
zoom_factor = tuple(np.array(target_shape) / np.array(cropped_data.shape))

# 进行重采样
resampled_data = zoom(cropped_data, zoom_factor, order=1)

# 创建重采样后的NIfTI影像对象
resampled_nii = nib.Nifti1Image(resampled_data, affine)

# 保存重采样后的NIfTI影像
nib.save(resampled_nii, 'resampled_image_1.nii')
fig1, axs1 = plt.subplots(1, 2, figsize=(10, 10))
axs1[0].imshow(img[:, :, 60], cmap='gray')
axs1[0].set_title('reshape')
axs1[1].imshow(resampled_data[:, :, 60], cmap='gray')
axs1[1].set_title('zoom')
# 可视化重采样后的影像
plt.axis('off')
plt.show()
