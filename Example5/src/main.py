import random

import torch
from torchvision import transforms, datasets
from torch.utils import data
from torchvision.transforms.functional import to_pil_image

import nn
from nn import Functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 调用GPU运算


class CNN:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1, True, device=device)
        self.pool1 = nn.MaxPool2d(stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 0, 1, True, device=device)
        self.pool2 = nn.MaxPool2d(stride=2)
        self.fc1 = nn.Linear(400, 120, device=device)
        self.fc2 = nn.Linear(120, 84, device=device)
        self.fc3 = nn.Linear(84, 10, device=device)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            for param in layer.params():
                params.append(param)
        return params


# 超参数
batch_size = 64  # 批量大小
learning_rate = 1e-4  # 学习率
epoch_num = 60  # 总训练轮数

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1317, 0.3094)])
MINIST_train = datasets.MNIST(root='../dataset', train=True, transform=transform, download=True)
MINIST_test = datasets.MNIST(root='../dataset', train=False, transform=transform, download=True)
dataset = data.DataLoader(MINIST_train, batch_size=batch_size, shuffle=True)
test_dataset = data.DataLoader(MINIST_test, batch_size=batch_size, shuffle=True)

model = CNN()  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 实例化损失函数
optimizer = nn.SGD(model.parameters(), lr=learning_rate)  # 实例化小批量随机梯度下降优化器

for epoch in range(epoch_num):
    run_loss = 0  # 每轮总损失
    iterator = int(len(dataset) / 10)
    for i, input in enumerate(dataset):
        x, y = input  # 从数据集中获取数据
        x, y = x.to(device), y.to(device)
        output = model(x)  # 前向传播
        optimizer.zero_grad()  # 清空梯度
        loss = criterion(output, y)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化
        run_loss += loss.sum().item()  # 计算每轮总损失
        if i >= iterator:
            print(f"{epoch + 1}/{epoch_num} {int(i / (len(dataset) - 1) * 100) + 1}% "
                  f"---------------------- loss: {run_loss / i / y.size(0)}")  # 输出每轮每10%的总损失
            iterator += int(len(dataset) / 10)


# 评估模型质量
# 这里我们采取通过计算有哪些输出是错误的来衡量。由于我们的test也是有噪声的，所以一个90%以上的精度的模型已经很好了
with torch.no_grad():
    det = 0
    total_count = 0
    for i, input in enumerate(test_dataset):
        x, y = input
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, pred = torch.max(output, dim=1, keepdim=True)
        pred = torch.flatten(pred)
        total_count += y.size(0)
        det += (y == pred).sum().item()
    det = det / total_count
    print(f"准确率:{det * 100}%")

    # 我们下面随机从测试集中抽一张图片来看一下预测是否正确
    img_idx = random.randint(0, len(MINIST_test) - 0)
    img = MINIST_test[img_idx][0].reshape(1, 1, 28, 28).to(device)
    output = model(img)
    print(output)
    output = F.softmax(output)
    value, pred = torch.max(output, dim=1, keepdim=True)
    print(f"这个张图片的数字是{pred.item()}, 它是{pred.item()}的概率为{value.item() * 100}%")
    img = img.reshape(1, 28, 28)
    img = to_pil_image(img)
    img.show()
