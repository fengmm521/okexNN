#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-24 09:29:05
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$
import tensorflow as tf
import json
# from numpy.random import RandomState


class TFNNTool(object):
	"""docstring for TFNNTool"""
	def __init__(self):
		# super(TFNNTool, self).__init__()
		self.batch_size = 8
		self.Weights = []
		self.Biases = []
		self.hidelayers = []
		self.inPlaceHolder = None
		self.outPlaceHolder = None
		self.flogPlaceHolder = None
		self.x = None
		self.y = None
		self.out = None
		self.cross_entropy = None
		self.train_step = None
		self.inSize = None  				#神经网络输入数据节点数
		self.middleSize = None    			#真对自编码器的中间输出节点数
		self.outSize = None  				#神经网络输出数据节点数
		self.nnSizes = [] 					#整个网络的各层节点数,从输入一直到输出

		self.inputDatas = []
		self.outDatas = []


	def createNewWeight(self,inSize,outSize,pStddev = 1,pSeed = 1):
		w = tf.Variable(tf.random_normal([inSize,outSize],stddev=pStddev,seed=pSeed))
		return w

	def createNewBiases(self,outSize):
		b = tf.Variable(tf.zeros([outSize]))
		return b

	def createPlaceHolder(self,dataType,pShape,pName): #tf.placeholder(tf.float32,shape=(None,2),name='x-input')
		p = tf.placeholder(dataType,shape,pName)
		return p

	#激活函数:activation_function:
	#http://blog.csdn.net/u011630575/article/details/78063641
	#tf.nn.sigmoid          输出为(0,1)     S(x) = 1/(1+e^(-x))
	#tf.nn.tanh  			输出为(-1,1)    tanh(x) = sinh(x)/cosh(x) = (e^x - e^(-x))/(e^x+e^(-x)) = (1-e^(-2x)/(1+e^(-2x))
	#tf.nn.relu	            输出为(0,++)    f(x) = max(x,0)
	#tf.nn.softplus         是relu的升级版   f(x) = log(1+e^x)
	#tf.nn.leakrelu         relu的另一版本   f(x) = max(x,leak*x)
	#tf.nn.elu				relu改进版		ELU(x) = {x    if x> 0,a(e^x - 1)  if x<= 0}
	#tf.nn.selu             relu另一改进版   SELU(x) = r{x   if x>0,a(e^x - 1)   if x<= 0}  且r > 1

	def add_Layer(self,inPuts,inSize,outSize,activation_function = None,wStddev = 1,wSeed = 1):
		w = self.createNewWeight(inSize,wStddev,wSeed)
		b = self.createNewBiases(outSize)
		Wx_plus_b = tf.matmul(inPuts, w) + b
		outputs = None
		if activation_function:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		self.Weights.append(w)
		self.Biases.append(b)
		self.hidelayers.append(outputs)
		return outputs
	#创建一个softmax神经网络
	def createSoftMaxNNWithNetSizes(self,nnSizes,activation_function = None):
		nnsizes = nnSizes
		self.nnSizes = nnSizes
		self.inSize = nnSizes[0]
		self.outSize = sizes[-1]
		afunc = activation_function
		if not afunc:
			afunc = tf.nn.relu
		countsizes = len(nnSizes)

		self.inPlaceHolder = self.createPlaceHolder(tf.float32,(None,self.inSize),'x-input')

		self.x = self.inPlaceHolder

		for i in range(len(nnsizes)):
			if i < countsizes - 3:
				sinput = nnsizes[i]
				soutput = nnsizes[i + 1]
				if not self.hidelayers:
					self.add_Layer(self.inPlaceHolder,sinput,soutput,afunc)
				else:
					self.add_Layer(self.hidelayers[-1],sinput,soutput,afunc)
			else:
				self.y = self.add_Layer(self.hidelayers[-1],sinput,soutput,tf.nn.softmax)

	#创建普通神经网络
	def createCommonNNWithNetSize(self,inputs,nnSizes,activation_function = None):

		inSize = nnSizes[0]
		outSize = nnSizes[-1]
		afunc = activation_function
		if not afunc:
			afunc = tf.nn.tanh
		countsizes = len(nnSizes)

		isFirst = True

		for i in range(len(nnSizes)):
			if i < countsizes - 2:
				sinput = nnsizes[i]
				soutput = nnsizes[i + 1]
				if isFirst:
					isFirst = False
					self.add_Layer(inputs,sinput,soutput,afunc)
				else:
					self.add_Layer(self.hidelayers[-1],sinput,soutput,afunc)
		return self.Weights,self.Biases,self.hidelayers

	def createEncodeSelfNNWithNetSize(self,enCodeSizes,middleSize,deCodeSizes,activation_function = None):
		self.inSize = enCodeSizes[0]
		self.middleSize = middleSize
		self.outSize = deCodeSizes[0]

		tfnnfunc = activation_function
		if not tfnnfunc:
			tfnnfunc = tf.nn.tanh

		self.inPlaceHolder = self.createPlaceHolder(tf.float32,(None,self.inSize),'x-input')
		self.x = self.inPlaceHolder

		ws,bs,hs = self.createCommonNNWithNetSize(self.inPlaceHolder, enCodeSizes,tfnnfunc)
		self.Weights = []
		self.Biases = []
		self.hidelayers = []

		self.out = self.add_Layer(hs[-1], middleSize, deCodeSizes[0])
		mw = self.Weights
		mb = self.Biases

		self.Weights = []
		self.Biases = []
		self.hidelayers = []
		dews,debs,dehs = self.createCommonNNWithNetSize(self.out, deCodeSizes,tfnnfunc)

		self.y = dehs[-1]
		self.Weights = ws + mw + dews
		self.Biases = bs + mb + debs

		self.hidelayers = hs + [self.out] + dehs[:-1]

		# self.flogPlaceHolder = self.createPlaceHolder(tf.float32,(None,self.outSize),'y-input')


	def setCross_entropy(self,pReduction_indices = [1]):
		self.flogPlaceHolder = self.createPlaceHolder(tf.float32, (None,self.outSize), 'y-input')
		self.cross_entropy = -tf.reduce_mean(tf.reduce_sum( self.flogPlaceHolder * tf.log(self.y),reduction_indices=pReduction_indices))

	def setTrain_step(self,batch_size = 8,learning_rate = 0.01):
		self.batch_size = batch_size
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)

	def training(self,X,Y,batch_size):
		pass

	def trainingTest(self):
		pass


	def initData(self,datapth):
		#10个5分钟数据，开，高，低，收，交易量
		f = open(datapth,'r')
		jstr = f.read()
		f.close()
		dats = json.loads(jstr)
		print len(dats)

	def saveOutData(self,savePth):
		pass

m5datapth = '../data/nndata/data5m_10.txt'

def main():
	nntool = TFNNTool()
	nntool.createEncodeSelfNNWithNetSize([120,60,32,8],2,[8,32,60,120],tf.nn.tanh)  #四层网络，有两个隐层
	nntool.setCross_entropy()
	nntool.setTrain_step(0.02)
	nntool.initData(m5datapth)
	# X = rdm.rand(dataset_size, 2)
	# Y = [[int(x1+x2 >= 0.8 and x1 + x2 <= 1.2),int(x1+x2>1.2 or x1 + x2 <0.8)] for (x1,x2) in X]

if __name__ == '__main__':
	main()

# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2 = tf.Variable(tf.random_normal([3,3],stddev=1,seed=1))
# w3 = tf.Variable(tf.random_normal([3,2],stddev=1,seed=1))

# b1 = tf.Variable(tf.zeros([3]))
# b2 = tf.Variable(tf.zeros([3]))
# b3 = tf.Variable(tf.zeros([2]))

# tf.nn.tanh

# x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# y_ = tf.placeholder(tf.float32,shape=(None,2),name='y-input')
# tf.nn.softplus
# a = tf.nn.relu(tf.matmul(x, w1) + b1)
# h = tf.nn.relu(tf.matmul(a, w2) + b2)
# y = tf.nn.softmax(tf.matmul(h,w3) + b3)

# cross_entropy = -tf.reduce_mean(tf.reduce_sum( y_ * tf.log(y),reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)


# #random data
# rdm = RandomState(1)
# dataset_size = 1280
# X = rdm.rand(dataset_size, 2)
# # Y = [[int(x1+x2 < 1),int(x1+x2>=1)] for (x1,x2) in X]

# Y = [[int(x1+x2 >= 0.8 and x1 + x2 <= 1.2),int(x1+x2>1.2 or x1 + x2 <0.8)] for (x1,x2) in X]

# rdm2 = RandomState(1)
# Xc = rdm2.rand(10, 2)

# #tensorflow session
# with tf.Session() as sess:
# 	init_op = tf.global_variables_initializer()
# 	sess.run(init_op)
# 	'''
# 	w1 = [[-0.81131822,1.48459876,0.06532937]]
# 			[-2.44270396,0.0992484,0.59122431]
# 	w2 = [[-0.81131822],[1.48459876],[0.06532937]]

# 	'''

# 	STEPS = 10000
# 	for i in range(STEPS):
# 		start = (i * batch_size) % dataset_size
# 		end = min(start + batch_size,dataset_size)
# 		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

# 		if i % 1000 == 0:
# 			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
# 			print "After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy)

# 	print 'w1',sess.run(w1)
# 	print 'w2',sess.run(w2)
# 	for d in Xc:
# 		xt = d
# 		xtinput = tf.constant(xt,shape=[1,2],dtype=tf.float32)
# 		a1 = tf.nn.relu(tf.matmul(xtinput, w1) + b1)
# 		h1 = tf.nn.relu(tf.matmul(a1, w2) + b2)
# 		y1 = tf.nn.softmax(tf.matmul(h1,w3) + b3)
# 		yt = sess.run(y1)
# 		yout = 0
# 		if yt[0][0] >= 0.5:
# 			yout = 1
# 		reldat = xt[0] + xt[1]
# 		outtmp =  int(xt[0] + xt[1] >= 0.8 and xt[0] + xt[1] <= 1.2)
# 		print xt,outtmp,reldat,yout,yt