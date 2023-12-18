import torch
from Dataset import Dataset
import nn

# 超参数
batch_size = 256  # 批量大小
learning_rate = 0.0001  # 学习率
epoch_num = 10000  # 总训练轮数

dataset = Dataset(root='../dataset/train/', batch_size=batch_size, shuffle=True)  # 读取数据集
linear = nn.Linear(4, 3)  # 实例化线性层
criterion = nn.CrossEntropyLoss()  # 实例化损失函数
optimizer = nn.SGD(linear.params(), lr=learning_rate)  # 实例化小批量随机梯度下降优化器

for epoch in range(epoch_num):
    run_loss = 0  # 每轮总损失
    for i, input in enumerate(dataset):
        x, y = input()  # 从数据集中获取数据
        output = linear(x)  # 前向传播
        optimizer.zero_grad()  # 清空梯度
        loss = criterion(output, y)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化
        run_loss += loss.sum().item()  # 计算每轮总损失
    run_loss /= len(dataset)
    print(f"{epoch + 1}/{epoch_num} ---------------------- loss: {run_loss}")  # 输出每轮总损失

# 评估模型质量
# 这里我们采取通过计算有哪些输出是错误的来衡量。由于我们的test也是有噪声的，所以一个90%以上的精度的模型已经很好了
with torch.no_grad():
    test_dataset = Dataset(root='../dataset/test/', batch_size=batch_size, shuffle=True)
    det = 0
    total_count = 0
    for i, input in enumerate(test_dataset):
        x, y = input()
        output = linear(x)
        y = torch.flatten(y)
        value, index = torch.max(output, dim=1, keepdim=True)
        for j, idx in enumerate(index):
            total_count += 1
            if idx != y[j]:
                det += 1
    det = (total_count - det) / total_count
    print(f"准确率:{det * 100}%")

    # 我们尝试调用以下模型来预测，比如输入一个label为0的数据，看看输出是不是0,再用softmax函数查看其概率
    test_tensor = torch.tensor([250., 0.7, 0.8, 11.])
    output = linear(test_tensor)
    output = nn.Functional.softmax(output)
    output = torch.flatten(output)
    value, index = torch.max(output, dim=0, keepdim=True)
    print(f"输入的是label为0的数据，输出的label是{index.item()}， 概率是{value.item() * 100}%")
