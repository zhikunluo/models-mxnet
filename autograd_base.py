from mxnet import autograd, nd 

x = nd.arange(4).reshape((4,1))
# apply memory to save gradient of x
x.attach_grad()
# ask MXNet to record the computation graph
with autograd.record():
	y = 2 * nd.dot(x.T, x)

# autograd
y.backward()

assert (x.grad - 4 * x).norm().asscalar() == 0

# predictiong mode and traning mode
print(autograd.is_training())
with autograd.record():
	print(autograd.is_training())

def f(a):
	b = a * 2
	while b.norm().asscalar() < 1000:
		b = b * 2
	if b.sum().asscalar() > 0:
		c = b
	else:
		c = 100 * b
	return c

a = nd.random.normal(shape=(1,))
a.attach_grad()
with autograd.record():
	c = f(a)
c.backward()

print(a.grad)

