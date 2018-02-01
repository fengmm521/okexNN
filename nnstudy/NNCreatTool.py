# View more python learning tutorial on my Youtube and Youku channel!!!

# My tutorial website: https://morvanzhou.github.io/tutorials/

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import numpy
# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

'''

'''

if not os.path.exists('nnsave'):
    os.mkdir('nnsave')



if not os.path.exists('../data/nnout'):
    os.mkdir('../data/nnout')

# Visualize encoder setting
# Parameters

class NNCreateTool(object):
    """docstring for NNCreateTool"""
    def __init__(self, enCodeSizes,middleSize,deCodeSizes,isReTrain = False,datapth = '../data/nndata/sigmoid_data5m_10.txt',savepth = 'nnsave/nn.txt'):
        self.enSizes = enCodeSizes
        self.middleSize = middleSize
        self.deSizes = deCodeSizes

        self.datapth = datapth
        self.savepth = savepth

        

        self.isInitFromFile = False
        if (not isReTrain) and os.path.exists(self.savepth):
            self.initNNFromFile()
            self.isInitFromFile = True
            

        self.inSize = self.enSizes[0]

        self.allSizes = self.enSizes + [self.middleSize] + self.deSizes
        self.middleIdx = len(self.enSizes)
        self.layerCount = len(self.allSizes)

        if self.isInitFromFile:
            self.X = tf.placeholder("float", [None, self.inSize])
            return

        self.weights = {}
        self.biases = {}

        

        self.enCodelayers = []
        self.deCodeLayers = []
        self.outLayer = None
        self.yLayer = None

        

        self.initWeights()
        self.initEncoder(self.X)
        self.initDecoder(self.outLayer)

        self.trData = self.initData()
        self.dataset_size = len(self.trData)
        self.cost = None
        self.optimizer = None

    def initData(self):
        f = open(self.datapth,'r')
        jstr = f.read()
        f.close()
        dats = json.loads(jstr)
        return dats

    def initNNFromFile(self):
        f = open(self.savepth,'r')
        jstr = f.read()
        f.close()
        inws = json.loads(jstr)
        ws = inws['w']
        bs = inws['b']

        self.middleSize = inws['mSize']
        self.enSizes = inws['enCodes']
        self.deSizes = inws['deCodes']

        self.weights = {}
        self.biases = {}

        print(len(ws),ws.keys())

        for n in range(len(ws)):
            w = ws[str(n)]
            self.weights[n] = tf.constant(numpy.array(w),dtype=tf.float32)

        for n in range(len(bs)):
            b = bs[str(n)]
            self.biases[n] = tf.constant(numpy.array(b),dtype=tf.float32)

    def initNNWithList(self,wlist,blist):
        ws = wlist
        bs = blist
        self.weights = {}
        self.biases = {}
        for n in range(len(ws)):
            w = ws[n]
            self.weights[n] = tf.constant(numpy.array(w),dtype=tf.float32)

        for n in range(len(bs)):
            b = bs[n]
            self.biases[n] = tf.constant(numpy.array(b),dtype=tf.float32)


    def getDataTypeIndex(self,data):
        if len(data) == self.inSize:
            x = tf.constant(data,shape=[1,self.inSize],dtype=tf.float32)
            self.initEncoder(x)

            outlist = []
            with tf.Session() as sess:
                out = sess.run(self.outLayer)
                outlist = out.tolist()
            return outlist[0]
        else:
            print('input data shap erro,input:%d'%(self.inSize))
            return None

    def getDatasTypeIndex(self,datas):
        if len(datas[0]) == self.inSize:

            x = tf.placeholder("float", [None, self.inSize])
            self.initEncoder(x)

            outlist = []
            with tf.Session() as sess:
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()

                for d in datas:
                    batch_xs = [d]
                    # print(len(batch_xs))
                    out = sess.run(self.outLayer, feed_dict={x: batch_xs})
                    # dattmp = out.tolist()[0]
                    # otmp0 = (dattmp[0] * 10000) - 6000
                    # otmp1 = (dattmp[1] * 10000) - 2500
                    outlist.append(out.tolist())
            return outlist
        else:
            print('input data shap erro,input:%d'%(self.inSize))
            return None

    def initWeights(self):
        
        for n in range(len(self.allSizes)):
            if n + 1 < self.layerCount:
                node1 = self.allSizes[n]
                node2 = self.allSizes[n + 1] 
                print(n,node1,node2)
                self.weights[n] = tf.Variable(tf.random_normal([node1, node2],stddev=1,seed=1))
                self.biases[n] = tf.Variable(tf.zeros([node2]))

    def initEncoder(self,x):

        layers = range(self.layerCount)
        self.enCodelayers = []

        for l in layers[:self.middleIdx]:
            if not self.enCodelayers:
                layertmp = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights[l]),self.biases[l]))
            else:
                layertmp = tf.nn.sigmoid(tf.add(tf.matmul(self.enCodelayers[-1], self.weights[l]),self.biases[l]))
            self.enCodelayers.append(layertmp)
        self.outLayer = self.enCodelayers[-1]


    def initDecoder(self,x):
        layers = range(self.layerCount - 1)
        print(layers)
        self.deCodeLayers = []
        for l in layers[self.middleIdx:]:
            if not self.deCodeLayers:
                layertmp = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights[l]),self.biases[l]))
            else:
                layertmp = tf.nn.sigmoid(tf.add(tf.matmul(self.deCodeLayers[-1], self.weights[l]),self.biases[l]))
            self.deCodeLayers.append(layertmp)
        self.yLayer = self.deCodeLayers[-1]

        
    def TrainingWithConfig(self,learning_rate = 0.0001,trStep = 5800,batch_size = 300,display_step = 100):
        if self.isInitFromFile:
            return
        self.training_epochs = trStep
        self.batch_size = batch_size
        self.display_step = display_step
        self.learning_rate = learning_rate

        y_true = self.X
        y_pred = self.yLayer


        # self.cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2.0))
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)


        self.cost = 0.5*tf.reduce_mean(tf.pow(tf.subtract(y_true,y_pred),2.0))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            # Training cycle

            for i in range(self.training_epochs):
                start = (i * self.batch_size) % self.dataset_size
                end = min(start + self.batch_size,self.dataset_size)

                batch_xs = self.trData[start:end]

                _, c = sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})

                if i % display_step == 0:
                    print("Epoch:", '%08d' % (i+1),
                          "cost=", "{:.9f}".format(c))
            print("Optimization Finished!")

            self.saveNNToFile(sess)

    def saveNNToFile(self,pSess):
        outws = {'w':{},'b':{}}
        for n in range(len(self.weights)):
            w = self.weights[n]
            wtmp = pSess.run(w)
            outws['w'][n] = wtmp.tolist()

        for n in range(len(self.biases)):
            b = self.biases[n]
            btmp = pSess.run(b)
            outws['b'][n] = btmp.tolist()

        # outws['sizes'] = self.allSizes
        outws['enCodes'] = self.enSizes
        outws['deCodes'] = self.deSizes
        outws['mSize'] = self.middleSize

        ostr = json.dumps(outws)
        f = open(self.savepth,'w')
        f.write(ostr)
        f.close()

        self.initNNWithList(outws['w'], outws['b'])

        print('nn save to:%s'%(self.savepth))

    def drawData(self):
        pass


def savelistToFileForJson(datas,savepth):
    ostr = json.dumps(datas)
    f = open(savepth,'w')
    f.write(ostr)
    f.close()


def savelistToFileForLines(datas,savepth):
    ostr = ''

    f = open(savepth,'w')

    for d in datas:
        ostr += str(d) + '\n'
    ostr = ostr[:-1]
    f.write(ostr)
    f.close()

def main():

    nnctool = NNCreateTool([50,120,60,32,10], 2, [10,32,60,120,50],isReTrain = False)
    tmpdatas = nnctool.initData()
    outs = nnctool.getDatasTypeIndex(tmpdatas)
    outdatas = []
    for d in outs:
        tmp0 = (d[0][0] * 10000) - 6000 - (11.45/2)
        tmp1 = (d[0][1] * 10000) - 2500 - (552/2)
        outdatas.append([tmp0,tmp1])
        

    
    savelistToFileForJson(outdatas, '../data/nnout/kline5out.json')
    savelistToFileForLines(outdatas, '../data/nnout/kline5lines.txt')

if __name__ == '__main__':
    main()

# dataset_size = len(datas)
      

# learning_rate = 0.0001    # 0.01 this learning rate will be better! Tested
# # Network Parameters
# n_input = 50  # MNIST data input (img shape: 28*28)

# # tf Graph input (only pictures)
# X = tf.placeholder("float", [None, n_input])

# # hidden layer settings
# #"klineS1":{"klineData":10,"layerEncode":[120,60,32,8],"mLayerNode":2,"layerDecode":[8,32,60,120],"state":"5minute","baseLine":1},
# n_hidden_1 = 120
# n_hidden_2 = 60
# n_hidden_3 = 32
# n_hidden_4 = 10
# n_hidden_5 = 2


# weights = {
#     'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=1,seed=1)),
#     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=1,seed=1)),
#     'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=1,seed=1)),
#     'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4],stddev=1,seed=1)),
#     'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5],stddev=1,seed=1)),

#     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4],stddev=1,seed=1)),
#     'decoder_h2': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3],stddev=1,seed=1)),
#     'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2],stddev=1,seed=1)),
#     'decoder_h4': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1],stddev=1,seed=1)),
#     'decoder_h5': tf.Variable(tf.random_normal([n_hidden_1, n_input],stddev=1,seed=1)),
# }

# biases = {
#     'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
#     'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
#     'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
#     'encoder_b4': tf.Variable(tf.zeros([n_hidden_4])),
#     'encoder_b5': tf.Variable(tf.zeros([n_hidden_5])),

#     'decoder_b1': tf.Variable(tf.zeros([n_hidden_4])),
#     'decoder_b2': tf.Variable(tf.zeros([n_hidden_3])),
#     'decoder_b3': tf.Variable(tf.zeros([n_hidden_2])),
#     'decoder_b4': tf.Variable(tf.zeros([n_hidden_1])),
#     'decoder_b5': tf.Variable(tf.zeros([n_input])),
# }



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

# def encoder(x):
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
#                                    biases['encoder_b1']))
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
#                                    biases['encoder_b2']))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
#                                    biases['encoder_b3']))
#     layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
#                                     biases['encoder_b4']))
#     layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
#                                     biases['encoder_b5'])
#     return layer_5


# def decoder(x):
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
#                                    biases['decoder_b1']))
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
#                                    biases['decoder_b2']))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
#                                 biases['decoder_b3']))
#     layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
#                                 biases['decoder_b4']))
#     layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
#                                 biases['decoder_b5']))
#     return layer_5


# datapth = '../data/nndata/sigmoid_data5m_10.txt'


# def initData(pth = datapth):
#     f = open(pth,'r')
#     jstr = f.read()
#     f.close()
#     dats = json.loads(jstr)
#     print(len(dats))
#     # self.inputDatas = dats
#     # self.outDatas = self.inputDatas
#     # print len(dats)
#     return dats

# datas = initData()

# print(len(datas[0]))

# # Construct model
# encoder_op = encoder(X)
# decoder_op = decoder(encoder_op)

# # Prediction
# y_pred = decoder_op
# # Targets (Labels) are the input data.
# y_true = X


# training_epochs = 3000
# batch_size = 300
# display_step = 100

# learning_rate = 0.0001    # 0.01 this learning rate will be better! Tested

# # Define loss and optimizer, minimize the squared error


# cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# # cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(y_pred,y_true),2.0))


# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# dataset_size = len(datas)

# # Launch the graph
# with tf.Session() as sess:
#     # tf.initialize_all_variables() no long valid from
#     # 2017-03-02 if using tensorflow >= 0.12
#     if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#         init = tf.initialize_all_variables()
#     else:
#         init = tf.global_variables_initializer()
#     sess.run(init)
#     # Training cycle

#     for i in range(training_epochs):
#         start = (i * batch_size) % dataset_size
#         end = min(start + batch_size,dataset_size)

#         batch_xs = datas[start:end]

#         _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

#         if i % display_step == 0:
#             print("Epoch:", '%08d' % (i+1),
#                   "cost=", "{:.9f}".format(c))

#     # for epoch in range(training_epochs):
#     #     # Loop over all batches
#     #     for i in range(total_batch):
#     #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
#     #         # Run optimization op (backprop) and cost op (to get loss value)
#     #         _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
#     #     # Display logs per epoch step
#     #     if epoch % display_step == 0:
#     #         print("Epoch:", '%04d' % (epoch+1),
#     #               "cost=", "{:.9f}".format(c))

#     print("Optimization Finished!")

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

