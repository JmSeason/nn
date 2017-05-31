import numpy as np
np.random.seed(1337)  # for reproducibility

from data_op import *
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

#load data
(x_train, y_train), (x_test, y_test) = get_mnist_download_data('datasets\\mnist.npz')

#数据预处理
x_train = x_train.reshape(-1, 28, 28)/255.
x_test = x_test.reshape(-1, 28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

TIME_STEPS = 28 # same as the height of the image
INPUT_SIZE = 28 # same as the width of the image
BATCH_SIZE = 40
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

#build model
model = Sequential()

#add rnn layer
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    unroll=True
))

#output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
for step in range(40000):
    x_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE]
    y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE]
    cost = model.train_on_batch(x_batch, y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX

    if step % 400 == 0:
        cost, accuracy = model.evaluate(x_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
