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
def load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return (x_train, y_train), (x_test, y_test)


(train_x, train_y), (test_x, test_y) = load_data("./dataset/mnist.npz")
# (train_x, train_y), (test_x, test_y) = mnist.load_data()
print("----------------------------")
print("Train shape:", train_x.shape)
print("Test shape:", test_x.shape)
print("----------------------------")

train_x = train_x.reshape([-1, 28, 28, 1]) / 255  # flatten 後歸一化


# 生成器 generator
g_sequential = Sequential(
    [
        Dense(7 * 7 * 64, input_shape=[100 + 10]),
        BatchNormalization(),
        LeakyReLU(),
        Reshape([7, 7, 64]),
        UpSampling2D([2, 2]),
        Conv2DTranspose(64, [3, 3], padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        UpSampling2D([2, 2]),
        Conv2DTranspose(1, [3, 3], padding="same", activation="sigmoid"),
    ]
)

# 判別器 discriminator
discriminator = Sequential(
    [
        Conv2D(64, [3, 3], padding="same", input_shape=[28, 28, 1]),
        BatchNormalization(),
        LeakyReLU(),
        MaxPool2D([2, 2]),
        Conv2D(64, [3, 3], padding="same"),
        BatchNormalization(),
        LeakyReLU(),
        MaxPool2D([2, 2]),
        Flatten(),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dense(11, activation="softmax"),
    ]
)

g_sample_input = Input([100])  # 生成器輸入
g_label_input = Input([], dtype="int32")  # 指定標籤輸入
x_input = Input([28, 28, 1])  # 真實樣本輸入
x_label_input = Input([], dtype="int32")  # 真實樣本標籤輸入

condition_g_sample_input = K.concatenate(
    [g_sample_input, K.one_hot(g_label_input, 10)]
)  # 合併隨機數據輸入與指定標籤獨熱碼

g_output = g_sequential(condition_g_sample_input)  # 生成器輸出
generator = Model(inputs=[g_sample_input, g_label_input], outputs=g_output)  # 生成器模型

# 裁減機率到區間 [1e-3, 1] 內，並求其 log ，避免 log 後為 inf，K.stop_gradient 表示訓練時不對其求梯度
#   這裡也可直接寫成 log_clip = Lambda(lambda x: K.log(x + 1e-3))
log_clip = Lambda(
    lambda x: K.log(K.clip(K.stop_gradient(x), 1e-3, 1) - K.stop_gradient(x) + x)
)

g_prob = discriminator(generator([g_sample_input, g_label_input]))  # 判別器識別假樣本的輸出
g_index = K.stack(
    [K.arange(0, K.shape(g_prob)[0]), g_label_input], axis=1
)  # 用於索引 g_prob 指定標籤機率值

d_prob = discriminator(x_input)  # 判別器識別真實樣本的輸出
x_index = K.stack(
    [K.arange(0, K.shape(d_prob)[0]), x_label_input], axis=1
)  # 用於索引 d_prob 正確標籤機率值


d_loss = -log_clip(tf.gather_nd(d_prob, x_index)) - log_clip(  # log(真實樣本正確標籤的機率值)
    1.0 - tf.gather_nd(g_prob, g_index)
)  # log(1-假樣本指定標籤的機率值)

fit_discriminator = Model(
    inputs=[g_sample_input, g_label_input, x_input, x_label_input], outputs=d_loss
)
fit_discriminator.add_loss(d_loss)  # 添加自定義loss
generator.trainable = False
for layer in generator.layers:
    if isinstance(layer, BatchNormalization):  # 設置 BatchNormalization 為訓練模式
        layer.trainable = True
fit_discriminator.compile(optimizer=Adam(0.001))
generator.trainable = True


g_loss = -log_clip(tf.gather_nd(g_prob, g_index))  # log(假樣本指定標籤的機率值)


fit_generator = Model(inputs=[g_sample_input, g_label_input], outputs=g_loss)
fit_generator.add_loss(g_loss)  # 添加自定義loss

# 生成器訓練時不更新discriminator的參數
discriminator.trainable = False
for layer in discriminator.layers:
    if isinstance(layer, BatchNormalization):  # 設置 BatchNormalization 為訓練模式
        layer.trainable = True
fit_generator.compile(optimizer=Adam(0.001))
discriminator.trainable = True

print("/////")
print("Discriminator summary:")
discriminator.summary()
print("/////")
print("Generator summary:")
generator.summary()
print("/////")

# train for 10000 times
batch_size = 64
for i in range(1):
    index = random.sample(range(len(train_x)), batch_size)
    x_label = train_y[index]
    x = train_x[index]
    g_sample = np.random.uniform(-1, 1, [batch_size, 100])
    g_label = np.random.randint(0, 10, [batch_size])

    fit_discriminator.fit(
        [K.constant(g_sample), K.constant(g_label), K.constant(x), K.constant(x_label)]
    )
    fit_generator.fit([K.constant(g_sample), K.constant(g_label)])

# save model
discriminator.save("./models/discriminator")
generator.save("./models/generator")
fit_discriminator.save("./models/fit_discriminator")
fit_generator.save("./models/fit_generator")

# image show
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(
            generator.predict(
                [K.constant(np.random.uniform(-1, 1, [1, 100])), K.constant([i])]
            )[0].reshape([28, 28]),
            cmap="gray",
        )
        axes[i, j].axis(False)
plt.show()
