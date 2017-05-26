import tensorflow as tf

def create_mlp_layer(input, in_size, out_size, activation_function=None):
    '''    
    :param input: 
    :param in_size: 
    :param out_size: 
    :param activation_function: 
    :return: 
    '''
    # create layer and return the Weights, biases, output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return Weights, biases, outputs

