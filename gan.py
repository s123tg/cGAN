import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from os import system

# data
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape([-1, 28 * 28]) / 255  # flatten后归一化

# 生成器
generator = Sequential([
    Dense(128, activation='relu', input_shape=[100]),
    Dense(28 * 28, activation='sigmoid')
])

# 判别器
discriminator = Sequential([
    Dense(128, activation='relu', input_shape=[28 * 28]),
    Dense(1, activation='sigmoid')
])

g_sample_input = Input([100])  # 生成器输入
x_input = Input([28 * 28])  # 真实数据输入

# 裁剪概率到区间[1e-6, 1]内，并求其log，避免log后为inf，K.stop_gradient表示训练时不对其求梯度
#   这里也可直接写成 log_clip = Lambda(lambda x: K.log(x + 1e-3))
log_clip = Lambda(lambda x: K.log(
    K.clip(K.stop_gradient(x), 1e-6, 1) - K.stop_gradient(x) + x))

g = discriminator(generator(g_sample_input))  # 假数据

# 判别器loss
d_loss = (
    - log_clip(discriminator(x_input))
    - log_clip(1.0 - g)
)

fit_discriminator = Model(
    inputs=[x_input, g_sample_input], outputs=d_loss)  # 训练discriminator所用模型
fit_discriminator.add_loss(d_loss)  # 添加自定义loss

# 在调用compile之前置generator.trainable为False，调用compile后的模型训练时不更新generator的参数
generator.trainable = False
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True

# 生成器loss
g_loss = (
    - log_clip(g)
)

fit_generator = Model(inputs=g_sample_input, outputs=g_loss)  # 训练generator所用模型
fit_generator.add_loss(g_loss)

# 生成器训练时不更新discriminator的参数
discriminator.trainable = False
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True


# train
batch_size = 64
for i in range(20000):
    # if i % 2000 == 0:
    # system("cls")
    #plt.imshow(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
    # plt.show()
    # print(i)
    # 随机选取batch_size个真样本
    x = train_x[random.sample(range(len(train_x)), batch_size)]
    # 生成batch_size个随机数据输入
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    # 训练辨别器，多输入需传入一个包含多个tensor的列表，此处用K.constant代替
    fit_discriminator.fit([K.constant(x), K.constant(g_sample)])
    fit_generator.fit(g_sample)  # 训练生成器


# image show
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(generator.predict(
            np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
        axes[i, j].axis(False)
plt.show()
