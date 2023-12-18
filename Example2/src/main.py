import torch
from Dataset import Dataset
import nn

# 超参数
batch_size = 256  # 批量大小
learning_rate = 0.00005  # 学习率
epoch_num = 4000  # 总训练轮数

dataset = Dataset(root='../dataset/train/', batch_size=batch_size, shuffle=True)  # 读取数据集
linear = nn.Linear(4, 2)  # 实例化线性层
criterion = nn.MSELoss()  # 实例化损失函数
optimizer = nn.SGD(linear.params(), lr=learning_rate)  # 实例化小批量随机梯度下降优化器

for epoch in range(epoch_num):
    run_loss = 0  # 每轮总损失
    for i, input in enumerate(dataset):
        x, y = input()  # 从数据集中获取数据
        output = linear(x)  # 前向传播
        optimizer.zero_grad()  # 清空梯度
        loss = criterion(output, y)  # 计算均方差损失
        loss.backward()  # 反响传播
        optimizer.step()  # 执行优化
        run_loss += loss.sum().item()  # 计算每轮总损失
    run_loss /= len(dataset)
    print(f"{epoch + 1}/{epoch_num} ---------------------- loss: {run_loss}")  # 输出每轮总损失

# 评估模型质量
with torch.no_grad():
    test_dataset = Dataset(root='../dataset/test/', batch_size=batch_size, shuffle=True)
    det = 0
    for i, input in enumerate(test_dataset):
        x, y = input()
        output = linear(x)
        det += abs((output - y) / y).sum() / y.numel()
        print(det)
    det /= len(test_dataset)
    print(f"精度:{(1 - det) * 100}%")

# 输出模型参数
print(linear.params())
