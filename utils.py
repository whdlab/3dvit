from matplotlib import pyplot as plt
from numpy import interp
from skimage import transform, exposure
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import numpy as np
import random
import torch

import data_add
from data_add import *

# 加载npy数据和label
from sklearn.metrics import roc_auc_score


def load_npy_data(data_dir, split):
    datanp = []  # images
    truenp = []  # labels
    for file in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        #         data[0][0] = resize(data[0][0], (224,224,224))
        if split == 'train':
            # 各种方式各扩充一次，共为原数据集大小的四倍
            data_sug = transform.rotate(data[0][0], 60)  # 旋转60度，不改变大小
            data_sug2 = exposure.exposure.adjust_gamma(data[0][0], gamma=0.5)  # 变亮
            # data_sug3 = data_add.randomflip(data[0][0])
            # data_sug4 = data_add.noisy(0.005)
            datanp.append(data_sug)
            truenp.append(data[0][1])
            datanp.append(data_sug2)
            truenp.append(data[0][1])
            # datanp.append(data_sug3)
            # truenp.append(data[0][1])
            # datanp.append(data_sug4)
            # truenp.append(data[0][1])

        datanp.append(data[0][0])
        truenp.append(data[0][1])
    datanp = np.array(datanp)
    # numpy.array可使用 shape。list不能使用shape。可以使用np.array(list A)进行转换。
    # 不能随意加维度
    datanp = np.expand_dims(datanp, axis=4)  # 加维度,from(256,256,128)to(256,256,128,1),according the cnn tabel.png
    datanp = datanp.transpose(0, 4, 1, 2, 3)
    truenp = np.array(truenp)
    print(datanp.shape, truenp.shape)
    # print(np.min(datanp), np.max(datanp), np.mean(datanp), np.median(datanp))
    return datanp, truenp


# def load_npy_data(data_dir, split, k=3):
#     data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
#     if split == 'train':
#         data_name = 'train_data.dat'
#         label_name = 'train_label.dat'
#     else:
#         data_name = 'val_data.dat'
#         label_name = 'val_label.dat'
#     datanp = np.memmap(data_name, dtype='float32', mode='w+', shape=(k * len(data_files), 1, 128, 128, 128))
#     truenp = np.memmap(label_name, dtype='int64', mode='w+', shape=(k * len(data_files),))
#
#     for i, file in enumerate(data_files):
#         data = np.load(file, allow_pickle=True)
#         if split == 'train':
#             data_sug = transform.rotate(data[0][0], 60)
#             data_sug2 = exposure.exposure.adjust_gamma(data[0][0], gamma=0.5)
#             datanp[3 * i, 0, :, :, :] = data_sug
#             datanp[3 * i + 1, 0, :, :, :] = data_sug2
#             truenp[3 * i] = data[0][1]
#             truenp[3 * i + 1] = data[0][1]
#             datanp[3 * i + 2, 0, :, :, :] = data[0][0]
#             truenp[3 * i + 2] = data[0][1]
#         else:
#             datanp[i, 0, :, :, :] = data[0][0]
#             truenp[i] = data[0][1]
#
#     return datanp, truenp


# 定义随机种子
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)


# 计算分类的各项指标
def calculate(score, label, th):
    score = np.array(score)
    label = np.array(label)
    pred = np.zeros_like(label)
    pred[score >= th] = 1
    pred[score < th] = 0

    TP = len(pred[(pred > 0.5) & (label > 0.5)])
    FN = len(pred[(pred < 0.5) & (label > 0.5)])
    TN = len(pred[(pred < 0.5) & (label < 0.5)])
    FP = len(pred[(pred > 0.5) & (label < 0.5)])

    AUC = metrics.roc_auc_score(label, score)
    result = {'AUC': AUC, 'acc': (TP + TN) / (TP + TN + FP + FN), 'sen': TP / (TP + FN + 0.0001),
              'spe': TN / (TN + FP + 0.0001)}
    #     print('acc',(TP+TN),(TP+TN+FP+FN),'spe',(TN),(TN+FP),'sen',(TP),(TP+FN))
    return result


def make_roc_pic(score, label, fpr, tpr, title='ROC'):
    AUC = metrics.roc_auc_score(label, score)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % AUC)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def make_rocs(roc_auc, fpr, tpr):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold  (AUC = %0.2f)' % roc_auc)


def make_train_pic(fold, acc_best, train_acc_list, train_losses_list, val_acc_list, val_loss=None,
                   save_path=None):
    # 绘图代码
    plt.figure()  # 创建新的图形窗口
    plt.plot(np.arange(len(train_losses_list)), train_losses_list, label="train loss")
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
    # plt.plot(np.arange(len(val_losses_list)), val_losses_list, label="valid loss")
    plt.plot(np.arange(len(val_acc_list)), val_acc_list, label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.title('Model accuracy&loss of fold{}, acc={:.4f}'.format(fold, acc_best))
    plt.savefig("./{}/fold{}_MOBILENETacc&loss.png".format(save_path, fold))
    # plt.show()
    plt.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model_super_state(save_root, model_path, name):
    path = os.path.join(save_root, name)
    net = torch.load(model_path, map_location=torch.device('cpu'))
    for key, value in net.items():
        if key != 'model_state_dict' and key != 'optimizer_state_dict':
            with open(path, 'w') as f:
                str = f'{key}: {value}\n'
                f.write(str)

        if key == 'optimizer_state_dict':
            opti = value['param_groups']
            with open(path, 'w') as f:
                str = f'{opti}\n'
                f.write(str)