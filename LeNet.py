import d2lzh as d2l
import mxnet as mx 
from mxnet import autograd, gluon, init, nd 
from mxnet.gluon import loss as gloss, nn 
import time

# define the LeNet
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Dense(120, activation='sigmoid'), 
	nn.Dense(84, activation='sigmoid'), 
	nn.Dense(10))

# data iterators
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# evaluate
def evaluate_accuracy(data_iter, net, ctx):
	acc_sum, n = nd.array([0], ctx=ctx), 0
	for X, y in data_iter:
		X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
		acc_sum += (net(X).argmax(axis=1) == y).sum()
		n += y.size
	return acc_sum.asscalar() / n 


# hyper parameters
lr, num_epochs = 0.9, 100

ctx = d2l.try_gpu()
# initialize
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
# train
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

