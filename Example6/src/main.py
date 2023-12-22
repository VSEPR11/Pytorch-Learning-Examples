import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from NN import CNN
from torchvision import transforms, datasets


# 超参数
batch_size = 64  # 批量大小
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 启用gpu，如果cuda不可用则用cpu
learning_rate = 0.001  # 学习率
epoch_num = 50  # 训练总轮数

# 导入数据集
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1, 0.5)])  # 定义组合变换
MNIST_train = datasets.MNIST('../dataset', train=True, transform=transforms, download=True)  # 训练集
MNIST_test = datasets.MNIST('../dataset', train=True, transform=transforms, download=True)  # 测试集
train_set = DataLoader(MNIST_train, batch_size=batch_size, shuffle=True)
test_set = DataLoader(MNIST_test, batch_size=batch_size, shuffle=True)

model = CNN(device)  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 对于分类问题，一般用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 实例化优化算法

writer = SummaryWriter('./logs/exp1')
model.train()

for epoch in range(epoch_num):
    run_loss = 0  # 训练总损失
    iterator = int(len(train_set) / 10)  # 打印迭代器
    i = 0
    for idx, data in enumerate(train_set):
        img, label = data  # 从数据中获取图像和标签
        img, label = img.to(device), label.to(device)  # 转换到device上执行
        output = model(img)  # 前向传播
        optimizer.zero_grad()  # 清空梯度
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化
        run_loss += loss.item()
        if idx >= iterator:  # 每轮执行10%，打印一次损失
            loss_10_percent = run_loss / len(train_set) * 10  # 百分之十轮的平均损失
            print(f"{(i + 1) * 10}%  epoch:{epoch + 1}/{epoch_num}  ------------------  "
                  f"loss:{loss_10_percent}")
            writer.add_scalar('loss', loss_10_percent, i + epoch * 10)  # 写入tensorboard
            run_loss = 0
            i += 1
            iterator += int(len(train_set) / 10)

writer.close()
torch.save(model.state_dict(), 'weight.pt')  # 保存模型
model.eval()

with torch.no_grad():
    det = 0
    total_count = 0
    for i, data in enumerate(test_set):
        img, label = data
        img, label = img.to(device), label.to(device)
        output = model(img)
        _, pred = torch.max(output, dim=1, keepdim=True)  # output中最大值的下标就是预测的标签
        pred = torch.flatten(pred)  # 展平pred
        total_count += img.size(0)
        det += (label == pred).sum().item()  # 统计label和pred相同元素的数量
    det = det / total_count
    print(f"准确率:{det * 100}%")
