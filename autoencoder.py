import numpy as np
np.random.seed(1337)  # for reproducibility

from data_op import *
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = get_mnist_download_data('datasets\\mnist.npz')

#预处理
x_train = x_train.astype('float32') / 255. - 0.5
x_test = x_test.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

encoding_dim = 2

INPUT_SIZE = x_train.shape[1]
input_img = Input(shape=(INPUT_SIZE, ))

#encoder layers
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

#decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(INPUT_SIZE, activation='tanh')(decoded)

#construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

#construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)

#compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

#training
autoencoder.fit(x_train, x_train,epochs=20, batch_size=256, shuffle=True)

#plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()