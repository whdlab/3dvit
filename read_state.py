import os

import torch
if __name__ == '__main__':
    path = './model_state/a.txt'
    model_pth = "cloud/vit128_save_t1_sag_jiaquan/model_t1_sag_5.6/t1_sag-fold1.pth"
    net = torch.load(model_pth, map_location=torch.device('cpu'))
    for key, value in net.items():
        if key != 'model_state_dict' and key != 'optimizer_state_dict':
            with open(path, 'a') as f:
                str = f'{key}: {value}\n'
                f.write(str)

        if key == 'optimizer_state_dict':
            opti = value['param_groups']
            with open(path, 'a') as f:
                str = f'{opti}\n'
                f.write(str)

def save_model_super_state(save_root, model_path, name):
    path = os.path.join(save_root, name)
    net = torch.load(model_path, map_location=torch.device('cpu'))
    for key, value in net.items():
        if key != 'model_state_dict' and key != 'optimizer_state_dict':
            with open(path, 'a') as f:
                str = f'{key}: {value}\n'
                f.write(str)

        if key == 'optimizer_state_dict':
            opti = value['param_groups']
            with open(path, 'a') as f:
                str = f'{opti}\n'
                f.write(str)