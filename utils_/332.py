import shutil

# k = "E:\\datasets\\new\\pMCI\\c\\S8908"
# i = "E:\\datasets\\new\\pMCI\\a\\S8908"
#
# moved_k = shutil.move(k, i)
# print("移动后的路径 k:", moved_k)

import os

directory_path = "E:\\datasets\\new\\pMCI\\c\\S8908"  # 目录的路径
parent_directory_path = os.path.dirname(directory_path)
print(parent_directory_path)