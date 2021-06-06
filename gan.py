import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

# data
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape([-1, 28 * 28]) / 255  # flatten 後歸一化

# 生成器 generator
generator = Sequential(
    [
        Dense(128, activation="relu", input_shape=[100]),
        Dense(28 * 28, activation="sigmoid"),
    ]
)

# 判別器 discriminator
discriminator = Sequential(
    [
        Dense(128, activation="relu", input_shape=[28 * 28]),
        Dense(1, activation="sigmoid"),
    ]
)

g_sample_input = Input([100])  # 生成器輸入
x_input = Input([28 * 28])  # 真實樣本輸入

# 裁減機率到區間 [1e-3, 1] 內，並求其 log ，避免 log 後為 inf，K.stop_gradient 表示訓練時不對其求梯度
#   這裡也可直接寫成 log_clip = Lambda(lambda x: K.log(x + 1e-3))
log_clip = Lambda(
    lambda x: K.log(K.clip(K.stop_gradient(x), 1e-6, 1) - K.stop_gradient(x) + x)
)

g = discriminator(generator(g_sample_input))  # 假數據

# 判別器 loss
d_loss = -log_clip(discriminator(x_input)) - log_clip(1.0 - g)

fit_discriminator = Model(
    inputs=[x_input, g_sample_input], outputs=d_loss
)  # 訓練 discriminator 所用模型
fit_discriminator.add_loss(d_loss)  # 添加自定義loss

# 在調用 compile 之前置 generator.trainable 為 False，調用 compile 後的模型訓練時不更新 generator 的參數
generator.trainable = False
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True

# 生成器 loss
g_loss = -log_clip(g)

fit_generator = Model(inputs=g_sample_input, outputs=g_loss)  # 訓練 generator 所用模型
fit_generator.add_loss(g_loss)

# 生成器訓練時不更新discriminator的參數
discriminator.trainable = False
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True


# train
batch_size = 64
for i in range(20000):
    # if i % 2000 == 0:
    # system("cls")
    # plt.imshow(generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]), cmap='gray')
    # plt.show()
    # print(i)
    # 隨機選取 batch_size 個真樣本
    x = train_x[random.sample(range(len(train_x)), batch_size)]
    # 生成 batch_size 個隨機數據輸入
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    # 訓練判別器，多輸入需傳入一個包含多個 tensor 的列表，此處用 K.constant 代替
    fit_discriminator.fit([K.constant(x), K.constant(g_sample)])
    fit_generator.fit(g_sample)  # 訓練生成器


# image show
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(
            generator.predict(np.random.uniform(-1, 1, [1, 100]))[0].reshape([28, 28]),
            cmap="gray",
        )
        axes[i, j].axis(False)
plt.show()
