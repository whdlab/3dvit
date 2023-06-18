from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

from data_add import transform_data
from model import LeNet
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import time
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional
from Dataset import CustomDataset
from Dataset import GetLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import mkdir, load_npy_data, calculate, _init_fn, set_seed, make_train_pic, \
    save_model_super_state
from convformer import Convformer
# from resnet import *
# from model import MobileNetV2
from vit3d import vit_base_patch16_224_in21k

set_seed(12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print(device)


class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            criterion,
            RESUME,
            start_model_save_epoch,
            init_lr,
            # scheduler
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = 0  # np.inf
        self.n_patience = 0
        self.lastmodel = None
        self.RESUME_path = RESUME
        self.init_lr = init_lr
        self.start_model_save_epoch = start_model_save_epoch
        # self.scheduler = scheduler

    def fit(self, epochs, train_loader, valid_loader, modility, model_root_save_path, floder_path, patience, fold):
        save_path = os.path.join(model_root_save_path, floder_path)
        mkdir(save_path)
        val_acc_list = []
        train_acc_list = []
        val_losses_list = []
        train_losses_list = []
        start_epoch = 0

        # 从上次的训练停止epoch开始训练
        if self.RESUME_path:
            checkpoint = torch.load(self.RESUME_path)  # 加载断点
            self.model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
            start_epoch = checkpoint['n_epoch']  # 设置开始的epoch
        # / self.scheduler.load_state_dict(checkpoint['lr_schedule'])

        for n_epoch in range(start_epoch + 1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_auc, train_time, rst_train, acc_train = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time, rst_val, acc_val = self.valid_epoch(valid_loader)

            # self.scheduler.step()

            train_acc_list.append(acc_train)  # for plot
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

            if (self.best_valid_score < valid_auc and n_epoch > self.start_model_save_epoch) or acc_val > acc_best:
                #             if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, modility, save_path, floder_path, current_val_loss=valid_loss, auc=valid_auc,
                                fold=fold,
                                all_epochs=epochs, patience=patience, current_val_acc=acc_val)
                self.info_message(
                    "AUC from {:.4f} to {:.4f}. ACC from {:.4f} to {:.4f}.Saved model to '{}'",
                    self.best_valid_score, valid_auc, acc_best, acc_val, self.lastmodel
                )
                self.best_valid_score = valid_auc
                self.n_patience = 0
                final_rst_train = rst_train
                final_rst_val = rst_val
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

        # final_rst_train = rst_train
        # final_rst_val = rst_val

        # 绘图代码
        make_train_pic(1, acc_best, train_acc_list, train_losses_list, val_acc_list,
                       folder_path=floder_path, save_path=save_path)

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
            loss = self.criterion(outputs, targets)  # weight=weight.to(self.device)
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

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)

    def save_model(self, n_epoch, modility, save_path, floder_path, current_val_loss, auc, fold, all_epochs, patience,
                    current_val_acc):
        os.makedirs(save_path, exist_ok=True)
        model_name = f"{modility}_{floder_path}-fold{fold}.pth"
        model_name_txt = model_name[:-4] + ".txt"
        self.lastmodel = os.path.join(save_path, model_name)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "next_epoch_best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                "auc": auc,
                "stop_model_checkpoint": self.lastmodel,
                "all_epoch": all_epochs,
                "init_lr": self.init_lr,
                "patience": patience,
                "current_val_acc": current_val_acc,
                "current_val_loss": current_val_loss,
                "space": '\n\n'
            },
            self.lastmodel,
        )
        # 保存参数txt
        save_model_super_state(save_path, self.lastmodel, model_name_txt)


def train_mri_type(mri_type, data_k_fold_path, model_save_path, model_floder, RESUME=None, start_model_save_epoch=None):
    rst_dfs = []
    train_dir = os.path.join(data_k_fold_path, 'train')
    val_dir = os.path.join(data_k_fold_path, 'test')

    if not start_model_save_epoch:
        start_model_save_epoch = 2

    train_data_retriever = CustomDataset(train_dir, split='train')
    valid_data_retriever = CustomDataset(val_dir, split='val')
    train_dataset_nums = len(train_data_retriever)
    val_dataset_nums = len(valid_data_retriever)

    epoch = 60
    patience = 10
    batch_size = 8
    model = Convformer(num_classes=1, has_logits=False)
    # pretrained_dict = torch.load("C:\\Users\\whd\\PycharmProjects\\AD&HC\\cloud\\3Dvit\\model_6.8_13.01"
    #                              "\\AD_VS_HC_model_6.8_13.01-fold1.pth")
    # # net_states = model.state_dict()
    # # state_dict = pretrained_dict['model_state_dict']
    #
    # del pretrained_dict['model_state_dict']['head.weight']
    # del pretrained_dict['model_state_dict']['head.bias']
    # model.load_state_dict(pretrained_dict['model_state_dict'], strict=False)
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))
    model.to(device)
    lr = 0.000002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #         optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch_functional.binary_cross_entropy_with_logits
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, int((epoch * 9) / 10), eta_min=1e-7, last_epoch=-1 )

    #         criterion = nn.BCELoss()
    print(f"nums of train datasets: {train_dataset_nums}\nnums of val datasets: {val_dataset_nums}")

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=_init_fn,
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever,
        batch_size=batch_size,  # SIZE=4, 8
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        worker_init_fn=_init_fn
    )

    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion,
        RESUME,
        start_model_save_epoch,
        init_lr=lr
        # scheduler=scheduler

    )

    rst = trainer.fit(
        epoch,
        train_loader,
        valid_loader,
        f"{mri_type}",
        model_save_path,
        model_floder,
        patience,
        fold=1,
    )
    rst_dfs.append(rst)

    rst_dfs = pd.concat(rst_dfs)
    print(rst_dfs)
    rst_dfs = pd.DataFrame(rst_dfs)
    rst_dfs.to_csv(os.path.join(model_save_path, model_floder, 'train_val_res_pf.csv'))  # 保存每一折的指标

    return trainer.lastmodel, rst_dfs


if __name__ == "__main__":
    target_classification = 0
    model_name = '3Dvit'
    model_floder = '5block_MS_shortcut1_model_6.15_13.01'
    target_name: str = ''
    if target_classification == 0:
        print("AD VS HC")
        target_name = "AD_VS_HC"
        datasets_root = "E:\\datasets\\using_datasets\\112all_ad&hc_npy_data\\togather_image_to_sub"
    elif target_classification == 1:
        print('sMCI VS pMCI')
        target_name = "sMCI_VS_pMCI"
        datasets_root = "E:\\datasets\\using_datasets\\112all_mci_npy_data\\togather_image_to_sub"
    elif target_classification == 2:
        print('sMCI VS HC')
        target_name = "sMCI_VS_HC"
        datasets_root = ""
    elif target_classification == 3:
        print('sMCI VS AD')
        target_name = "sMCI_VS_AD"
        datasets_root = ""
    elif target_classification == 4:
        print('pMCI VS HC')
        target_name = "pMCI_VS_HC"
        datasets_root = ""
    elif target_classification == 5:
        print('pMCI VS AD')
        target_name = "pMCI_VS_HC"
        datasets_root = ""

    model_root_path = f'cloud/{model_name}'
    modelfiles, rst_dfs = train_mri_type(target_name, datasets_root, model_root_path, model_floder, RESUME=None)
