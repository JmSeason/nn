import layer_creater as lc
import neural_network_creater as nnc
import numpy as np
import tensorflow as tf

#训练数据准备
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] #(300, 1)
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#定义数据节点
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#定义神经网络层结构
layer_creaters = []
def lc1():
    Weight1, biase1, output1 = lc.create_mlp_layer(xs, 1, 10, activation_function=tf.nn.relu)
    return output1

layer_creaters.append(lc1)

def lc2(input):
    Weight2, biase2, output2 = lc.create_mlp_layer(input, 10, 1)
    return output2

layer_creaters.append(lc2)

prediction = nnc.mlp_create(layer_creaters)

#定义 loss 表达式
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 1))

#选择 optimizer 使 loss 达到最小
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#对所有 tf 变量进行初始化
init = tf.global_variables_initializer() #有变量(tf.Variable)就要初始化
sess = tf.Session()

#上面的定义都没有运算，直到 sess.run 才会运算
sess.run(init)

#迭代多次学习，sess.run optimizer
for i in range(1000):
    #training train_step 和 loss 都是由 placeholder 定义的运算，所以要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))