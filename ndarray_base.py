from mxnet import nd 

# basis of ndarray
# 1. create ndarrays  
x = nd.arange(12)
#print("the shape of x: %d" % x.shape)
#print("The size of x: %d" % x.size)
X = x.reshape((3,4))
Z = nd.zeros((2,3,4))
O = nd.ones((3,4))
Y = nd.array([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
R = nd.random.normal(0, 1, shape=(3,4))

# 2. operate ndarrays
Add = X + Y
Mul = X * Y
Div = X / Y
Exp = Y.exp() # same as np.exp(Y)
Dot = nd.dot(X, Y.T) 
CCT_0, CCT_1 = nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1)
Eq = X == Y
Sum = X.sum() # same as np.sum(X)
Tran_scal = X.norm().asscalar()

# 3. broadcasting
A = nd.arange(3).reshape((3,1))
B = nd.arange(2).reshape((1,2))
broad_add = A + B

# 4. index
sub_array = X[1:3] 
X[1,2] = 9
X[1:2, :] = 12

# 5. memory 
before = id(Y)
Y = Y + X
id(Y) == before

Z = Y.zeros_like()
before = id(Z)
Z[:] = X + Y
id(Z) == before

nd.elemwise_add(X, Y, out=Z)
id(Z) == before

before = id(X)
X[:] = X + Y # or X += Y
id(X) == before

# 6. ndarray & numpy
import numpy as np 
P = np.ones((2,3))
D = nd.array(P)
P = D.asnumpy()

