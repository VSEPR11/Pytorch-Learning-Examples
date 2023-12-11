import os
import random
import numpy as np
import torch


# 数据类，用来存放每一组的数据
class data:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, *args, **kwargs):
        return self.x, self.y


# 数据集类，用来读取数据集的数据。
# root为根目录，batch_size为批量大脚，shuffle是是否打乱读入的数据。
# 我这里是在初始化时就将所有数据读如，实际上应该是在迭代过程中才将数据读入，初始化时只存储所有数据的目录
# 这个类不用读懂，不重要
class Dataset:
    def __iter__(self):
        return iter(self.data_list)

    def __init__(self, root: str, batch_size=1, shuffle=False):
        self.data_list = []
        data_dirs = os.listdir(root + 'data/')
        label_dirs = os.listdir(root + 'label/')
        x_lines = []
        y_lines = []
        for i, item in enumerate(data_dirs):
            x = []
            with open(root + 'data/' + data_dirs[i]) as file:
                x_string = file.readline().replace('\n', ' ').split(' ')
                for string in x_string:
                    if len(string) > 0:
                        x.append(float(string))
            y = []
            with open(root + 'label/' + label_dirs[i]) as file:
                y_string = file.readline().replace('\n', ' ').split(' ')
                for string in y_string:
                    if len(string) > 0:
                        y.append(float(string))
            x_lines.append(x)
            y_lines.append(y)
        if shuffle:
            while len(x_lines) > 0:
                if len(x_lines) >= batch_size:
                    idx = random.sample(np.arange(0, len(x_lines)).tolist(), batch_size)
                else:
                    idx = np.arange(0, len(x_lines))
                idx.sort()
                x_batch = []
                y_batch = []
                for i in idx:
                    x_batch.append(x_lines[i])
                    y_batch.append(y_lines[i])
                self.data_list.append(data(torch.tensor(x_batch), torch.tensor(y_batch)))
                for i, num in enumerate(idx):
                    num -= i
                    del x_lines[num]
                    del y_lines[num]
        else:
            x_batch = []
            y_batch = []
            counter = 0
            length = len(x_lines)
            for i in range(length - 1, -1, -1):
                x_batch.append(x_lines[i])
                y_batch.append(y_lines[i])
                del x_lines[i]
                del y_lines[i]
                counter += 1
                if len(x_lines) >= batch_size == counter:
                    self.data_list.append(data(torch.tensor(x_batch), torch.tensor(y_batch)))
                    counter = 0
            if len(x_batch) > 0:
                self.data_list.append(data(torch.tensor(x_batch), torch.tensor(y_batch)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
