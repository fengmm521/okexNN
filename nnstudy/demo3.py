# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

'''

'''



# Visualize encoder setting
# Parameters


# Network Parameters
n_input = 50  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
#"klineS1":{"klineData":10,"layerEncode":[120,60,32,8],"mLayerNode":2,"layerDecode":[8,32,60,120],"state":"5minute","baseLine":1},
n_hidden_1 = 120
n_hidden_2 = 60
n_hidden_3 = 32
n_hidden_4 = 10
n_hidden_5 = 2

# weights = {
#     'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
#     'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
#     'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
#     'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
#     'encoder_h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5],)),

#     'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_4],)),
#     'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
#     'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
#     'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
#     'decoder_h5': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
# }
# biases = {
#     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
#     'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
#     'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),

#     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
#     'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
#     'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
#     'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
#     'decoder_b5': tf.Variable(tf.random_normal([n_input])),
# }
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=1,seed=1)),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=1,seed=1)),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=1,seed=1)),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=1,seed=1)),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5],stddev=1,seed=1)),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4],stddev=1,seed=1)),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3],stddev=1,seed=1)),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2],stddev=1,seed=1)),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],stddev=1,seed=1)),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, n_input],stddev=1,seed=1)),
}

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.zeros([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.zeros([n_hidden_5])),

    'decoder_b1': tf.Variable(tf.zeros([n_hidden_4])),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b4': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b5': tf.Variable(tf.zeros([n_input])),
}



# def encoder(x):
#     layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
#                                    biases['encoder_b1']))
#     layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
#                                    biases['encoder_b2']))
#     layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
#                                    biases['encoder_b3']))
#     layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
#                                     biases['encoder_b4']))
#     layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
#                                     biases['encoder_b5'])
#     return layer_5


# def decoder(x):
#     layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
#                                    biases['decoder_b1']))
#     layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#                                    biases['decoder_b2']))
#     layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
#                                 biases['decoder_b3']))
#     layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
#                                 biases['decoder_b4']))
#     layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
#                                 biases['decoder_b5']))
#     return layer_5

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4']))
    layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                                    biases['encoder_b5'])
    return layer_5


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                biases['decoder_b5']))
    return layer_5


datapth = '../data/nndata/sigmoid_data5m_10.txt'


def initData(pth = datapth):
    f = open(pth,'r')
    jstr = f.read()
    f.close()
    dats = json.loads(jstr)
    print(len(dats))
    # self.inputDatas = dats
    # self.outDatas = self.inputDatas
    # print len(dats)
    return dats

datas = initData()

print(len(datas[0]))

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


training_epochs = 3000
batch_size = 300
display_step = 100

learning_rate = 0.0001    # 0.01 this learning rate will be better! Tested

# Define loss and optimizer, minimize the squared error


cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(y_pred,y_true),2.0))


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


dataset_size = len(datas)

# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # Training cycle

    for i in range(training_epochs):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)

        batch_xs = datas[start:end]

        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        if i % display_step == 0:
            print("Epoch:", '%08d' % (i+1),
                  "cost=", "{:.9f}".format(c))

    # for epoch in range(training_epochs):
    #     # Loop over all batches
    #     for i in range(total_batch):
    #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    #     # Display logs per epoch step
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch+1),
    #               "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # # Applying encode and decode over test set
    # encode_decode = sess.run(
    #     y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # plt.show()

    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()

