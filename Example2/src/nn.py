import torch


# 自定义线性层类
# input_dim输入维度，output_dim输出维度(指输入的列数和输出的行数）
# forward() 为前向传播，计算 x dot w + b的值
# params 返回参数列表
class Linear:
    def __init__(self, input_dim, output_dim):
        self.w = torch.randn(size=(input_dim, output_dim), requires_grad=True)
        self.b = torch.randn(size=(1, output_dim), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def params(self):
        return [self.w, self.b]


# 自定义随机梯度下降的优化器
# params模型参数， lr 学习率 learning_rate
# zero_grad() 用于清空张量的梯度(张量的梯度是需要手动清零的，不然会影响下一轮的迭代）
# step() 用于将反向传播获取的梯度优化到参数上
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for i in range(len(self.params)):
            if self.params[i].grad is not None:
                self.params[i].grad.zero_()

    def step(self):
        for i in range(len(self.params)):
            with torch.no_grad():
                self.params[i] -= self.params[i].grad * self.lr


# 自定义均方差损失函数，用来计算损失，注意这里需要求和，因为只有0维的张量才能调用backward()函数
class MSELoss:
    def __call__(self, pred, label):
        return 1 / 2 / len(label) * ((pred - label) ** 2).sum()
