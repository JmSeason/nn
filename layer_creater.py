import tensorflow as tf

def create_mlp_layer(in_size, out_size, activation_function=None):
    '''
    :param in_size: 
    :param out_size: 
    :param activation_function: 
    :return: 
    '''
    # create layer and return the Weights, biases, output of this layer
    xs = tf.placeholder(tf.float32, [None, 1])
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(xs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return xs, Weights, biases, outputs

