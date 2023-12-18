import random
import numpy as np


path = './test/'
data_path = path + 'data/'
label_path = path + 'label/'

total_num = 2000
iterator = 0

# 分有两个标签 0, 1，表示温度正常和温度异常
# 温度的生成范围在30至45，我们认为36至37是正常体温

noise_rate = 0.01  # 有0.01的数据是标注错误的

for i in range(total_num):
    y = random.randint(0, 1)
    x = random.uniform(30, 45)
    if y == 0:
        x = random.uniform(36, 37)
    if random.uniform(0, 1) < noise_rate:
        if y == 0:
            y = 1
        else:
            y = 0
    with open(data_path + str(iterator) + ".txt", 'w') as file:
        file.write(str(x))
        file.close()
    with open(label_path + str(iterator) + ".txt", 'w') as file:
        file.write(str(y))
        file.close()
    iterator += 1
