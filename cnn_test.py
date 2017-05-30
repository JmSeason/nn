import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

def get_mnist_download_data(path):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

batch_size = 32
#获取训练\测试数据
(x_train, y_train), (x_test, y_test) = get_mnist_download_data('datasets\\mnist.npz')

#数据预处理
x_train = x_train.reshape(-1, 1, 28, 28)/255.
x_test = x_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#build cnn
model = Sequential()

#conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(batch_size, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))
model.add(Activation('relu'))

#pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

#conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

#pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

#Fully connected layer 1 input shape (64 * 7 * 7) = 3136, output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

#another way to define your optimizer
adam = Adam(lr=1e-4)

#add metric to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('training ----------------')
#way to train model
model.fit(x_train, y_train, epochs=1, batch_size=batch_size)

print('testing -----------------')
#evaluate the model with the metric we defined earlier
loss, accuracy = model.evaluate(x_test[0:320], y_test[0:320]) #批量要是32的倍数，有点奇怪，后续探讨

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)