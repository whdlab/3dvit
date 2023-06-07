import nibabel as nib
import os
import numpy as np
from skimage.transform import resize
import pandas as pd
net_data = []
img_data = nib.load(os.path.join('C:\\Users\\whd\\Desktop/wm_941_S_1363.nii'))
img = img_data.get_fdata()
img = resize(img, (128, 128, 128), order=0)  # 将图像大小进行统一缩放，方便输入网络，分别为（h,w,c）,可根据自己的数据集来更改
img = np.array(img)

label_data = 0
net_data.append([img, label_data])
np.save('C:\\Users\\whd\\Desktop/ADNI1_Screening_1.5T/sd.npy', net_data)  # 保存