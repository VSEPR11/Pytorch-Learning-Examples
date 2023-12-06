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
epoch_num = 1000  # 训练论数

model = Model()
train_loader = data.DataLoader(Dataset('../dataset/train/data/', '../dataset/train/label/'),
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(Dataset('../dataset/test/data/', '../dataset/test/label/'),
                              batch_size=batch_size,
                              shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learn_rate)

for epoch in range(epoch_num):
    run_loss = 0
    for i, input_data in enumerate(train_loader):
        x, y = input_data
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
    print(f"{epoch + 1} / {epoch_num} ------------------- loss: {run_loss}")

print('\nTrain End')

model.eval()

with torch.no_grad():
    det = 0
    for data in test_loader:
        x, y = data
        output = model(x)
        for i in range(len(y)):
            det += abs(y[i] - output[i]).item()
        det /= len(y)
    print(f'准确率: {(1 - det / len(train_loader)) * 100}%')

    for param in model.parameters():
        print(param.data)
