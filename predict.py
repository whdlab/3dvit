from matplotlib import pyplot as plt
from numpy import interp
from model import MobileNetV2
from model import LeNet
from skimage import transform, exposure
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc

from resnet import resnet34, resnet18
from utils import mkdir, load_npy_data, calculate, _init_fn, set_seed, make_roc_pic
from Dataset import GetLoader

test_path = 'E:\\datasets\\data_forlenet\\split_forad\\test'
model_path = '.\\cloud\\MOBILENET_105x125x105_save_t1_sag_jiaquan'
model_floder = 'model_t1_sag_4.8'  # model_TSC3:reset the train and test as machine learning
save_path = os.path.join(model_path, model_floder)
mkdir(save_path)

datanp_test, truenp_test = load_npy_data(test_path, '1')
# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
test_data_retriever = GetLoader(datanp_test, truenp_test)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(modelfile, mri_type, fold_num):
    print(modelfile)
    data_retriever = test_data_retriever
    data_loader = torch_data.DataLoader(
        data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # model = resnet18()
    model = MobileNetV2(2)
    model.to(device)

    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred = []
    ids = []
    y_all = []

    for e, batch in enumerate(data_loader, 1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch[0].to(device))).cpu().numpy().squeeze()
            #             tmp_pred = model(batch[0].to(device)).cpu().numpy().squeeze()
            targets = batch[1].to(device)
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())

            y_all.extend(batch[1].tolist())

    y_all = [1 if x > 0.5 else 0 for x in y_all]
    auc = roc_auc_score(y_all, y_pred)
    fpr_micro, tpr_micro, th = metrics.roc_curve(y_all, y_pred)
    # make_roc_pic(y_pred, y_all, fpr_micro, tpr_micro, title=fold_num)
    max_th = -1
    max_yd = -1
    for i in range(len(th)):
        yd = tpr_micro[i] - fpr_micro[i]
        if yd > max_yd:
            max_yd = yd
            max_th = th[i]

    rst_val = pd.DataFrame([calculate(y_pred, y_all, max_th)])
    preddf = pd.DataFrame({"label": y_all, "y_pred": y_pred})
    return preddf, rst_val, y_all, y_pred # 返回每一个病人的预测分数的表格和预测指标AUC，ACC，Sep，sen


# save_path = os.path.join(model_path, model_floder)

# 加载保存的5个模型
modelfiles = []
for model_name in os.listdir(save_path):
    if model_name.endswith('.pth'):
        model_path_final = os.path.join(save_path, model_name)
        modelfiles.append(model_path_final)

# 1，用5个模型对测试集数据进行预测，并取平均值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rst_test_all = []
scores = []
df_test = {}
fold_nums=len(modelfiles)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for m in modelfiles:
    mtype = m.split('/')[-1].split('-')[1].split('.')[0]
    preddf, rst_test, y_all, y_pred = predict(m, 'T1', 'test_{}'.format(mtype))

    fpr, tpr, thresholds = roc_curve(y_all,  y_pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = roc_auc_score(y_all, y_pred)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC %s (AUC = %0.2f)' % (mtype, roc_auc))

    rst_test_all.append(rst_test)
    df_test[mtype] = preddf["y_pred"]

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of 3-fold in test set')
plt.legend(loc="lower right")
plt.savefig("{}/ROC.png".format(save_path))
plt.show()


rst_test_all = pd.concat(rst_test_all)
rst_test_all = pd.DataFrame(rst_test_all)
df_test['label'] = preddf["label"]
df_test = pd.DataFrame(df_test)
rst_test_all.loc['mean'] = rst_test_all.mean(axis=0)
rst_test_all.to_csv(os.path.join(save_path, 'test_res_pf.csv'))
print('测试集{}折模型预测，取平均指标：{.4f}'.format(fold_nums, rst_test_all))

# 2，对5折预测的分数取平均值，并计算指标
df_test = pd.DataFrame(df_test)
df_test["Average"] = 0
# for mtype in mri_types:
for i in range(0, 3):
    df_test["Average"] += df_test.iloc[:, i]
df_test["Average"] /= 3
df_test.to_csv(os.path.join(save_path, 'test_score.csv'))
auc = roc_auc_score(df_test["label"], df_test["Average"])
print(f"test ensemble AUC: {auc:.4f}")
fpr_micro, tpr_micro, th = metrics.roc_curve(df_test["label"], df_test["Average"])
max_th = -1
max_yd = -1
for i in range(len(th)):
    yd = tpr_micro[i] - fpr_micro[i]
    if yd > max_yd:
        max_yd = yd
        max_th = th[i]
print(max_th)
# make_roc_pic(df_test["Average"], df_test["label"], fpr_micro, tpr_micro, title='Test ensemble ROC curve')    # 绘制roc
rst_test = pd.DataFrame([calculate(df_test["Average"], df_test["label"], max_th)])
rst_test.to_csv(os.path.join(save_path, 'test_ensembel_res.csv'))
print('{}折分数取平均之后的测试集指标：{}'.format(fold_nums, rst_test))
print('{}折预测的分数，以及分数平均值表格：{}'.format(fold_nums, df_test))
