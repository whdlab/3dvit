import os
import shutil

# 遍历adfold和hcfold文件夹，获取每个折文件夹的路径
adfold_path = 'E:\\datasets\\3dvit\\togather_image_to_sub\\adfold'
hcfold_path = 'E:\\datasets\\3dvit\\togather_image_to_sub\\hcfold'

adfold_dirs = [os.path.join(adfold_path, d) for d in os.listdir(adfold_path) if
               os.path.isdir(os.path.join(adfold_path, d))]
hcfold_dirs = [os.path.join(hcfold_path, d) for d in os.listdir(hcfold_path) if
               os.path.isdir(os.path.join(hcfold_path, d))]

for i in range(len(adfold_dirs)):
    adfold_dir = adfold_dirs[i]
    hcfold_dir = hcfold_dirs[i]
    fold_list = [adfold_dir, hcfold_dir]
    # 在该折文件夹内创建一个新文件夹，用于存放该折中ad和hc类别同一折的前缀文件夹
    fold_root = 'E:\\datasets\\3dvit\\togather_image_to_sub\\fold'
    if not os.path.exists(fold_root):
        os.makedirs(fold_root)
    fold_path = os.path.join(fold_root, 'fold_f{}/'.format(i + 1))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    file = open(os.path.join(fold_root, "fold_images_imfo.txt"), "a+")
    all_ad_hc_list = []
    # 遍历adfold和hcfold文件夹，将同一折的前缀文件夹分别复制到新文件夹中
    for fold_dir in fold_list:
        count_cla = 0
        for prefix in os.listdir(fold_dir):
            prefix_path = os.path.join(fold_dir, prefix)
            if os.path.isdir(prefix_path):
                new_prefix_path = os.path.join(fold_path, prefix)
                if not os.path.exists(new_prefix_path):
                    os.makedirs(new_prefix_path)
                for npy_file in os.listdir(prefix_path):
                    npy_file_path = os.path.join(prefix_path, npy_file)
                    shutil.copy(npy_file_path, new_prefix_path)
                    count_cla += 1
        all_ad_hc_list.append(count_cla)
        print("Class {} had splited:{}".format(fold_dir[-13:-11], count_cla))
        file.write("Class {} had splited:{}\n".format(fold_dir[-13:-11], count_cla))
    print("fold {} over, sum of images: {}".format(fold_dir[-1:], sum(all_ad_hc_list)))
    file.write("fold {} over, sum of images: {}\n".format(fold_dir[-1:], sum(all_ad_hc_list)))
    file.write("fold {} over\n".format(fold_dir[-1:]))
    file.close()
