import random

import numpy as np
from torchvision.transforms import transforms


class randomflip180(object):
    def __call__(self, data):
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
            data = np.flip(data, axis=2)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
            data = np.flip(data, axis=0)
        return data.copy()


class randomflip(object):
    def __call__(self, data):
        # print(data.shape)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=0)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=1)
        if random.uniform(0, 1) > 0.5:
            data = np.flip(data, axis=2)
        return data.copy()


class noisy(object):
    def __init__(self, radio):
        self.radio = radio

    def __call__(self, data):
        _, l, w, h = data.shape
        num = int(l * w * h * self.radio)
        for _ in range(num):
            x = np.random.randint(0, l)
            y = np.random.randint(0, w)
            z = np.random.randint(0, h)
            data[0,x, y, z] = data[0,x, y, z] + np.random.uniform(0, self.radio)
        return data


transform_data = transforms.Compose([
    randomflip(),
    randomflip180(),
    noisy(0.001)])
