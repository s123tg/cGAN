##
# @filename: cgan.py
# @date: 21/5/22

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


# load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape([-1, 28, 28, 1]) / 255  # flatten后归一化


# 生成器 generator
g_sequential = Sequential([
    Dense(7 * 7 * 64, input_shape=[100 + 10]),
    BatchNormalization(),
    LeakyReLU(),
    Reshape([7, 7, 64]),
    UpSampling2D([2, 2]),
    Conv2DTranspose(64, [3, 3], padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    UpSampling2D([2, 2]),
    Conv2DTranspose(1, [3, 3], padding='same', activation='sigmoid')
])

# 判别器 discriminator
discriminator = Sequential([
    Conv2D(64, [3, 3], padding='same', input_shape=[28, 28, 1]),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool2D([2, 2]),
    Conv2D(64, [3, 3], padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    MaxPool2D([2, 2]),
    Flatten(),
    Dense(128),
    BatchNormalization(),
    LeakyReLU(),
    Dense(11, activation='softmax')
])

g_sample_input = Input([100])  # 生成器输入
g_label_input = Input([], dtype='int32')  # 指定标签输入
x_input = Input([28, 28, 1])  # 真实样本输入
x_label_input = Input([], dtype='int32')  # 真实样本标签输入

condition_g_sample_input = K.concatenate(
    [g_sample_input, K.one_hot(g_label_input, 10)])  # 合并随机数据输入与指定标签独热码

g_output = g_sequential(condition_g_sample_input)  # 生成器输出
generator = Model(inputs=[g_sample_input, g_label_input],
                  outputs=g_output)  # 生成器模型

# 裁剪概率到区间[1e-3, 1]内，并求其log，避免log后为inf，K.stop_gradient表示训练时不对其求梯度
#   这里也可直接写成 log_clip = Lambda(lambda x: K.log(x + 1e-3))
log_clip = Lambda(lambda x: K.log(
    K.clip(K.stop_gradient(x), 1e-3, 1) - K.stop_gradient(x) + x))

g_prob = discriminator(
    generator([g_sample_input, g_label_input]))  # 判别器识别假样本的输出
g_index = K.stack([K.arange(0, K.shape(g_prob)[0]),
                  g_label_input], axis=1)  # 用于索引g_prob指定标签概率值

d_prob = discriminator(x_input)  # 判别器识别真实样本的输出
x_index = K.stack([K.arange(0, K.shape(d_prob)[0]),
                  x_label_input], axis=1)  # 用于索引d_prob正确标签概率值


d_loss = (
    - log_clip(tf.gather_nd(d_prob, x_index))  # log(真实样本正确标签概率值)
    - log_clip(1.0 - tf.gather_nd(g_prob, g_index))  # log(1-假样本指定标签的概率值)
)

fit_discriminator = Model(
    inputs=[g_sample_input, g_label_input, x_input, x_label_input], outputs=d_loss)
fit_discriminator.add_loss(d_loss)  # 添加自定义loss
generator.trainable = False
for layer in generator.layers:
    if isinstance(layer, BatchNormalization):  # 设置BatchNormalization为训练模式
        layer.trainable = True
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True


g_loss = (
    -log_clip(tf.gather_nd(g_prob, g_index))  # log(假样本指定标签的概率值)
)


fit_generator = Model(inputs=[g_sample_input, g_label_input], outputs=g_loss)
fit_generator.add_loss(g_loss)  # 添加自定义loss

# 生成器训练时不更新discriminator的参数
discriminator.trainable = False
for layer in discriminator.layers:
    if isinstance(layer, BatchNormalization):  # 设置BatchNormalization为训练模式
        layer.trainable = True
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True


# train for 10000 times
batch_size = 64
for i in range(10000):
    # if i % 10 == 0:
    #     clear_output()
    #     plt.imshow(generator.predict([K.constant(
    #         np.random.uniform(-1, 1, [1, 100])), K.constant([i % 10])])[0].reshape([28, 28]), cmap='gray')
    #     plt.title(str(i % 10))
    #     plt.show()
    # print(i)
    index = random.sample(range(len(train_x)), batch_size)
    x_label = train_y[index]
    x = train_x[index]
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    g_label = np.random.randint(0, 10, [batch_size])

    fit_discriminator.fit([K.constant(g_sample), K.constant(
        g_label), K.constant(x), K.constant(x_label)])
    fit_generator.fit([K.constant(g_sample), K.constant(g_label)])

# image show
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(generator.predict(
            np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
        axes[i, j].axis(False)
plt.show()
