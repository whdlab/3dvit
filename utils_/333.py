import glob
import os
import re
import shutil

source_dir = "E:/datasets/new/pMCI/newpmci"
loss_dir = 'E:\\datasets\\new\\pMCI\\80loss'

nii_files = glob.glob(os.path.join(source_dir, "**/S*"), recursive=True)
list = os.listdir(loss_dir)
f = 0
for i in nii_files:
    j = i.split('\\')
    k = j[-1]
    if k in list:
        loss_k = os.path.join(loss_dir,k)
        if os.path.exists(os.path.dirname(i)):
            shutil.rmtree(os.path.dirname(i))
        shutil.move(loss_k, os.path.dirname(i))
        f += 1
print("移动完成")
print(f)


# # 遍历源文件夹中的每个子文件夹
# for root, dirs, files in os.walk(source_dir):
#     for dir_name in dirs:
#         dir_path = os.path.join(root, dir_name)
#         for i in os.listdir(dir_path):
#             dir_path1 = os.path.join(root, dir_name, i)
#             for j in os.listdir(dir_path1):
#                 dir_path2 = os.path.join(root, dir_name, i, j)
#                 for k in os.listdir(dir_path2):
#                     dir_path3 = os.path.join(root, dir_name, i, j, k)
#             # 检查子文件夹是否不包含 "mri" 文件夹
#                     if "mri" not in os.listdir(dir_path3):
#                         # 将子文件夹移动到目标文件夹
#                         shutil.move(dir_path3, target_dir)
#                         os.makedirs(target_dir, exist_ok=True)
#                         print(f"移动子文件夹 {dir_name} 到目标文件夹")
#
# print("移动完成")
