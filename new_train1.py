from matplotlib import pyplot as plt
from data_add import transform_data
from model import LeNet
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
from Dataset import CustomDataset
from Dataset import GetLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import mkdir, load_npy_data, calculate, _init_fn, set_seed

# from resnet import *
# from model import MobileNetV2
from vit3d import vit_base_patch16_224_in21k

set_seed(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_path = 'E:\\datasets\\3dvit\\train'
model_path = 'cloud/vit128_save_t1_sag_jiaquan'
model_floder = 'model_t1_sag_5.6'
save_path = os.path.join(model_path, model_floder)
mkdir(save_path)

# datanp_train, truenp_train = load_npy_data(train_path, 'train')
# datanp_val, truenp_val = load_npy_data(val_path, 'val')

# # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
# train_data_retriever2 = GetLoader(datanp_train, truenp_train)


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion,
            RESUME
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = 0  # np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.RESUME_path = RESUME

    def fit(self, epochs, train_loader, valid_loader, modility, save_path, patience, fold):
        global acc_avg
        best_auc = 0
        val_acc_list = []
        train_acc_list = []
        val_losses_list = []
        train_losses_list = []
        start_epoch = 0

        if self.RESUME_path:
            checkpoint = torch.load(self.RESUME_path)  # 加载断点
            self.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
            start_epoch = checkpoint['n_epoch']  # 设置开始的epoch
            # lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        for n_epoch in range(start_epoch + 1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_auc, train_time, rst_train, acc_train = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time, rst_val, acc_val = self.valid_epoch(valid_loader)

            train_acc_list.append(acc_train)    # for plot
            val_acc_list.append(acc_val)
            train_losses_list.append(train_loss)
            val_losses_list.append(valid_loss)

            # 计算验证集上的平均验证准确度
            acc_avg = sum(val_acc_list) / len(val_acc_list)
            acc_best = max(val_acc_list)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f},time: {:.2f} s ",
                n_epoch, train_loss, train_auc, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            #             if True:
            # if self.best_valid_score > valid_loss:

            if self.best_valid_score < valid_auc and n_epoch > 10:
                #             if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, modility, save_path, valid_loss, valid_auc, fold)
                self.info_message(
                    "loss decrease from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_auc, self.lastmodel
                )
                self.best_valid_score = valid_auc
                self.n_patience = 0

            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

        final_rst_train = rst_train
        final_rst_val = rst_val

        # 绘图代码
        plt.plot(np.arange(len(train_losses_list)), train_losses_list, label="train loss")
        plt.plot(np.arange(len(train_acc_list)), train_acc_list, label="train acc")
        # plt.plot(np.arange(len(val_losses_list)), val_losses_list, label="valid loss")
        plt.plot(np.arange(len(val_acc_list)), val_acc_list, label="valid acc")
        plt.legend()  # 显示图例
        plt.xlabel('epoches')
        plt.title('Model accuracy&loss of fold{}, avg_acc={:.4f},best_acc={:.4f}'.format(fold, acc_avg, acc_best))
        plt.savefig("./{}/fold{}_MOBILENETacc&loss.png".format(save_path, fold))
        plt.show()

        all_rst = [final_rst_train, final_rst_val]
        rst = pd.concat(all_rst, axis=1)
        print(rst)

        print('fold ' + str(fold + 1) + ' finished!')
        return rst

    def train_epoch(self, train_loader):

        self.model.train()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []


        for step, batch in enumerate(train_loader, 1):
            X = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            # w = [1.788, 1.]  # 标签0和标签1的权重
            # weight = torch.zeros(targets.shape)  # 权重矩阵
            # weight.to(self.device)
            # for i in range(targets.shape[0]):
            #     weight[i] = w[int(targets[i])]
            loss = self.criterion(outputs, targets)   # weight=weight.to(self.device)
            loss.to(device)
            loss.backward()

            sum_loss += loss.detach().item()
            y_all.extend(batch[1].tolist())
            outputs_all.extend(outputs.tolist())

            self.optimizer.step()

            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, outputs_all)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]
        acc = calculate(outputs_all, y_all, max_th)['acc']
        rst_train = pd.DataFrame([calculate(outputs_all, y_all, max_th)])

        return sum_loss / len(train_loader), auc, int(time.time() - t), rst_train, acc

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch[1].tolist())
                outputs_all.extend(outputs.tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss / step, end="\r")

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)
        fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, outputs_all)
        max_th = -1
        max_yd = -1
        for i in range(len(th)):
            yd = tpr_micro[i] - fpr_micro[i]
            if yd > max_yd:
                max_yd = yd
                max_th = th[i]
        acc = calculate(outputs_all, y_all, max_th)['acc']
        rst_val = pd.DataFrame([calculate(outputs_all, y_all, max_th)])

        return sum_loss / len(valid_loader), auc, int(time.time() - t), rst_val, acc

    def save_model(self, n_epoch, modility, save_path, loss, auc, fold):

        os.makedirs(save_path, exist_ok=True)
        self.lastmodel = os.path.join(save_path, f"{modility}-fold{fold}.pth")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                "auc": auc,
                "stop_model_checkpoint": self.lastmodel
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


def train_mri_type(mri_type, data_k_fold_path, RESUME=None):
    # 5折交叉验证，每一折保存一个模型
    rst_dfs = []
    fold_dirs = ['f1', 'f2', 'f3']
    fold_path = [os.path.join(data_k_fold_path, f) for f in fold_dirs]

    for fold, fold_dir in enumerate(fold_path):
        print(f'---------Fold {fold} start!-----------:')
        # 训练集文件夹路径
        train_dir = os.path.join(data_k_fold_path, fold_dir, 'train')
        # 验证集文件夹路径
        val_dir = os.path.join(data_k_fold_path, fold_dir, 'val')

        train_data_retriever = CustomDataset(train_dir, split='train')
        valid_data_retriever = CustomDataset(val_dir, split='val')

        train_loader = torch_data.DataLoader(
            train_data_retriever,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=_init_fn,
        )

        valid_loader = torch_data.DataLoader(
            valid_data_retriever,
            batch_size=4,   # SIZE=4, 8
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=_init_fn
        )

        model = vit_base_patch16_224_in21k(num_classes=1, has_logits=False)
        # model = resnet18()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        #         optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch_functional.binary_cross_entropy_with_logits
        #         criterion = nn.BCELoss()
        trainer = Trainer(
            model,
            device,
            optimizer,
            criterion,
            RESUME
        )

        rst = trainer.fit(
            100,
            train_loader,
            valid_loader,
            f"{mri_type}",
            save_path,
            100,
            fold,
        )
        rst_dfs.append(rst)

    rst_dfs = pd.concat(rst_dfs)
    print(rst_dfs)
    rst_dfs = pd.DataFrame(rst_dfs)
    rst_dfs.to_csv(os.path.join(save_path, 'train_val_res_pf.csv'))  # 保存每一折的指标

    return trainer.lastmodel, rst_dfs


modelfiles = None
# 开始训练
if not modelfiles:
    root = "E:\\datasets\\3dvit\\togather_image_to_sub\\fold\\fold_result"
    modelfiles, rst_dfs = train_mri_type('t1_sag', root, RESUME=None)
    print(modelfiles)
