import nibabel as nib
import os
import numpy as np
import torch
from skimage.transform import resize

model_pth = "C:\\Users\\whd\\PycharmProjects\\AD&HC\\cloud\\3Dvit\\model_6.8_13.01\\AD_VS_HC_model_6.8_13.01-fold1.pth"
net = torch.load(model_pth, map_location=torch.device('cpu'))
print()