import random
import numpy as np


path = './train/'
data_path = path + 'data/'
label_path = path + 'label/'

total_num = 10000

mu = 0  # 高斯噪声均值
sigma = 30  # 高斯噪声标准差
max_x = sigma * 10  # x最大值
min_x = sigma * -10  # x最小值
iterator = 0  # 迭代器，用来标注文件名

# 我们假设 y = xw * b中 w, b为以下值
w = np.array([[1., 2.],
              [3., 4.],
              [5., 6.],
              [7., 8.]])
b = np.array([9., 10.])

for i in range(total_num):
    x = []
    x_string = ''
    for j in range(4):
        num = random.uniform(min_x, max_x)  # 随机生成均匀分布的x
        x.append(num)
        x_string += str(num)
        if j != 3:
            x_string += ' '
    X = np.array(x)
    with open(data_path + str(iterator) + ".txt", 'w') as file:
        file.write(x_string)
        file.close()
    y = X.dot(w) + b + np.array([random.gauss(mu, sigma), random.gauss(mu, sigma)])  # 加上高斯噪声
    y_string = str(y[0]) + ' ' + str(y[1])
    with open(label_path + str(iterator) + '.txt', 'w') as file:
        file.write(y_string)
        file.close()
    iterator += 1
