import torch
import os
from torch import nn
from torch.utils import data
from torch import optim


# 定义模型，由于只是一个简单的线性关系，只有一层线性层
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# 自定义数据集类，用以导入自动生成的数据集
class Dataset(data.Dataset):
    def __init__(self, train_path: str, label_path: str):
        self.x = self.__read_txt__(train_path)
        self.y = self.__read_txt__(label_path)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def __read_txt__(self, path: str):
        files_path = os.listdir(path)
        data_list = []
        for file_path in files_path:
            line = []
            with open(path + file_path) as file:
                strings = file.readline().replace('\n', '').split(' ')
                for string in strings:
                    if len(string) > 0:
                        line.append(float(string))
                file.close()
            data_list.append(line)
        return torch.tensor(data_list)


batch_size = 128  # 批量大小
learn_rate = 0.00000005  # 学习率
epoch_num = 100  # 训练论数

model = Model()  # 实例化模型
# 初始化训练集和测试集
train_loader = data.DataLoader(Dataset('../dataset/train/data/', '../dataset/train/label/'),
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(Dataset('../dataset/test/data/', '../dataset/test/label/'),
                              batch_size=batch_size,
                              shuffle=True)
# 定义损失函数，本次使用最简单的均方差损失函数
criterion = nn.MSELoss()
# 定义优化其，本次使用随机梯度下降算法
optimizer = optim.SGD(model.parameters(), lr=learn_rate)

for epoch in range(epoch_num):  # 训练epoch_num次
    run_loss = 0  # 累计损失，用来衡量每一轮的优化程度
    for i, input_data in enumerate(train_loader):
        x, y = input_data
        optimizer.zero_grad()  # 梯度清零
        output = model(x)  # 前向传播
        loss = criterion(output, y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化
        run_loss += loss.item()  # 损失累加
    print(f"{epoch + 1} / {epoch_num} ------------------- loss: {run_loss / batch_size}")

print('\nTrain End')

model.eval()

with torch.no_grad():  # 禁用梯度

    # 计算误差，通过预期和实际值的差值的累加的均值来衡量
    det = 0
    for data in test_loader:
        x, y = data
        output = model(x)
        for i in range(len(y)):
            det += abs(y[i] - output[i]).item()
        det /= len(y)
    print(f'准确率: {(1 - det / len(train_loader)) * 100}%')

    # 输出模型参数，由于线性层基本上就是 y = wT * x + b， 所以可以将第一个tensor视为拟合结果，可以看出，拟合是很好的
    for param in model.parameters():
        print(param.data)
