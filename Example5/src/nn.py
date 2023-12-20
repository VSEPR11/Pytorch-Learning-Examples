import torch
import torch.nn.functional as f
from abc import abstractmethod


# 自定义线性层类
# input_dim输入维度，output_dim输出维度(指输入的列数和输出的行数）
# forward() 为前向传播，计算 x dot w + b的值
# params 返回参数列表
class Linear:
    def __init__(self, input_dim, output_dim, device=torch.device('cpu')):
        self.w = torch.randn(size=(input_dim, output_dim), requires_grad=True, device=device)
        self.b = torch.randn(size=(1, output_dim), requires_grad=True, device=device)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def params(self):
        return [self.w, self.b]


# 自定义卷积层，调用torch.nn.functional.conv2d来完成
class Conv2d:
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 bias=False,
                 device=torch.device('cpu')):
        self.kernel = torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size),
                                  requires_grad=True,
                                  device=device)  # 自定义卷积核
        self.padding = padding
        self.stride = stride
        self.bias = bias
        if self.bias:
            self.b = torch.randn(out_channels, requires_grad=True, device=device)  # 自定义偏置
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return f.conv2d(x, self.kernel, self.b, stride=self.stride, padding=self.padding)

    def params(self):
        if self.bias:
            return [self.kernel, self.b]
        else:
            return [self.kernel]


# 自定义池化层，调用max_pool2d_with_indices
class MaxPool2d:
    def __init__(self, kernel_size=2, padding=0, stride=1):
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (padding, padding)
        self.stride = (stride, stride)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return f.max_pool2d_with_indices(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)[0]


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


# 函数类
class Functional:
    # 静态函数softmax，第一行的目的是为了防止出现inf,因为e的10次方都太大了，更何况输入可能大于10。
    # 我们采取每一个x都减去其每一行的最大值，这样x中的数值一定小于等于0，而且这样操作后结果和没有这一行代码的结果是一样的
    @staticmethod
    def softmax(x):
        x = x - torch.max(x, dim=1, keepdim=True)[0]  # 转化x防止inf
        x_exp = torch.exp(x)  # 下面是softmax的公式
        x_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_prob = x_exp / x_sum
        return x_prob

    # 将softmax和torch.log相结合
    @staticmethod
    def log_softmax(x):
        x = x - torch.max(x, dim=1, keepdim=True)[0]
        x_sum = torch.sum(torch.exp(x), dim=1, keepdim=True)
        x = x - torch.log(x_sum + 1e-8)  # 加1e-8防止出现log(0)导致inf
        return x

    # 将[C - 1]格式的label转化成独立热编码
    @staticmethod
    def one_hot(x, class_num):
        x = x.long()
        x = x.unsqueeze(-1)
        device = x.device
        one_hot = torch.zeros(*x.shape[:-1], class_num, device=device)
        one_hot.scatter_(-1, x, 1)
        return one_hot

    @staticmethod
    def relu(x):
        zeros = torch.zeros_like(x)
        x = torch.max(x, zeros)
        return x

    @staticmethod
    def sigmoid(x):
        x = 1 / (1 + torch.exp(-x))
        return x

    @staticmethod
    def tanh(x):
        x = (1 - torch.exp(-2 * x)) / (1 + torch.exp(-2 * x))
        return x


# 自定义交叉熵损失函数，前两个if是当输入的不是独立热编码的时候的输入，而第三个是当是独立热编码的输入。
# 为了规范运算，我们将所有的输入都转换成独立热编码
class CrossEntropyLoss:
    def __call__(self, input, target):
        if len(target.size()) == 1 or target.size()[1] != input.size()[1]:  # 当输入不是独立热编码的时候，我们要将其转化成独立热编码
            target = torch.flatten(target)
            target = Functional.one_hot(target, input.size()[1])
        log_prob = Functional.log_softmax(input) * target  # 将原来的torch.log(softmax)换成log_softmax来规避上溢和下溢
        loss = -log_prob
        return loss.sum() / len(target)
