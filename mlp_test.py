import layer_creater as lc
import neural_network_creater as nnc
import numpy as np
import tensorflow as tf

#训练数据准备
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] #(300, 1)
noise = np.random.normal(0, 0.05, x_data.shape)