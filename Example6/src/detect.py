import torch
import random
from NN import CNN
from torchvision import datasets, transforms


model = CNN()  # 构建模型
model.load_state_dict(torch.load('./weight.pt'))  # 导入模型权重
model.eval()

test = datasets.MNIST('../dataset', train=True, transform=transforms.ToTensor(), download=True)
img, label = test[random.randint(0, len(test) - 1)]  # 从MNIST测试集中随机抽一场图片来预测

output = model(img)  # 执行预测
_, predict = torch.max(output, dim=1, keepdim=True)  # 最大值即为结果
print(f'预测这张图片是{predict.item()},实际上图片为{label}')

# 展示这张图片
to_pil = transforms.ToPILImage()
image = to_pil(img)
image.show()
