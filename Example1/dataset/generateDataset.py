import os
import random

# 假设 y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * [x0, x1, x2, ..., x9].T()

label_path = './test/label/'
data_path = './test/data/'

data_num = 1000

random_range = 6  # 正常数据的随机偏差范围

# 噪声数据的概率和随机偏差范围
noise_possi = 0.05
noise_range_max = 50
noise_range_min = 30
iterator = 0
for i in range(data_num):
    # 如果是正常数据
    if random.uniform(0, 1) > noise_possi:
        y = 0
        line = ''
        for idx in range(0, 10):
            x = random.uniform(0, 2000)  # x0到x9取值在0到2000范围内
            line += str(x)
            if idx != 9:
                line += ' '
            y += x * (idx + 1)
        y += random.uniform(-random_range / 2, random_range / 2)
        with open(data_path + str(iterator) + '.txt', mode='w') as file:
            file.write(line)
            file.close()
        with open(label_path + str(iterator) + '.txt', mode='w') as file:
            file.write(str(y))
            file.close()
    # 如果是噪声数据
    else:
        y = 0
        line = ' '
        for idx in range(0, 10):
            x = random.uniform(0, 2000)
            line += str(x)
            if idx != 9:
                line += ' '
            y += x * (idx + 1)
        if random.randint(0, 2) == 0:
            y -= random.uniform(noise_range_min, noise_range_max + 1)
        else:
            y += random.uniform(noise_range_min, noise_range_max + 1)
        with open(data_path + str(iterator) + '.txt', mode='w') as file:
            file.write(line)
            file.close()
        with open(label_path + str(iterator) + '.txt', mode='w') as file:
            file.write(str(y))
            file.close()
    iterator += 1
