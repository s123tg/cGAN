import keras
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


# number = input("Please input a test number:")
generator = keras.models.load_model("./models/generator")

# image show
'''
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
'''
num = float(input("input:"))
plt.imshow(generator.predict([K.constant(np.random.uniform(-1, 1, [1, 100])), K.constant([num])])[0].reshape([28, 28]), cmap='gray')
plt.title(str(int(num)))
plt.show()