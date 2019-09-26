def net(x, hidden_sizes=(64,64), activation=None, 
         output_activation=None):   
    for h_units in hidden_sizes[:-1]:
        x = tf.layers.dense(x, h_units, activation=activation)

    return tf.layers.dense(x, hidden_sizes[-1], activation=output_activation)

def q_net(x, a, hidden_sizes=(128,128), activation=tf.nn.relu, output_activation=None):
    return tf.squeeze(net(tf.concat([x,a], 1), list(hidden_sizes)+[1], activation, output_activation))


def policy_net(x, act_dim, hidden_sizes=(128,128), activation=tf.nn.relu, output_activation=tf.tanh):
    return net(x, list(hidden_sizes)+[act_dim], activation, output_activation)