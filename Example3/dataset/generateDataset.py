import random
import numpy as np


path = './train/'
data_path = path + 'data/'
label_path = path + 'label/'

total_num = 10000
iterator = 0

# 分有三个标签，0,1,2, 三个标签分布均匀
#         x0          x1         x2         x3
# 0   200 - 300   0.6 - 1.0   0.7 - 1.2   8 - 12
# 1   150 - 250   0.4 - 0.6   0.3 - 0.5   13 - 16
# 2   100 - 200   0.3 - 0.5   0.6 - 0.9   4 - 7

noise_rate = 0.01  # 有0.01的数据是标注错误的

for i in range(total_num):
    label = random.randint(0, 2)
    if label == 0:
        x0 = random.uniform(200, 300)
        x1 = random.uniform(0.6, 1)
        x2 = random.uniform(0.7, 1.2)
        x3 = random.randint(8, 12)
    elif label == 1:
        x0 = random.uniform(150, 250)
        x1 = random.uniform(0.4, 0.6)
        x2 = random.uniform(0.3, 0.5)
        x3 = random.randint(13, 16)
    else:
        x0 = random.uniform(100, 200)
        x1 = random.uniform(0.3, 0.5)
        x2 = random.uniform(0.6, 0.9)
        x3 = random.randint(4, 7)
    if random.uniform(0, 1) < noise_rate:  # 如果是标注错误的数据，我们就将label随机改成其他label
        label_other = random.randint(0, 2)
        while label_other == label:
            label_other = random.randint(0, 2)
        label = label_other

    with open(data_path + str(iterator) + ".txt", 'w') as file:
        file.write(str(x0))
        file.write(' ')
        file.write(str(x1))
        file.write(' ')
        file.write(str(x2))
        file.write(' ')
        file.write(str(x3))
        file.close()
    with open(label_path + str(iterator) + ".txt", 'w') as file:
        file.write(str(label))
    iterator += 1
