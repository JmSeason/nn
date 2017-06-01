import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

def get_train_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (BATCH_SIZE, TIME_STEPS)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

#build model
model = Sequential()

#build LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
    output_dim=CELL_SIZE,
    return_sequences=True, #True, output at all steps. False, output at last step
    stateful=True         #True: the final state of batch1 is feed into the initial state of batch2
))

#add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

adam = Adam(LR)
model.compile(optimizer=adam, loss='mse')

print('Training --------------')
for step in range(500):
    x_batch, y_batch, xs = get_train_batch()
    cost = model.train_on_batch(x_batch, y_batch)
    pred = model.predict(x_batch, BATCH_SIZE)
    plt.plot(xs[0, :], y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)
