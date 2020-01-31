import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
	range=(np.sqrt(6))/(np.sqrt(in_size+out_size))

	W = np.random.uniform(-range,range, size=(in_size,out_size))
	b = np.zeros(out_size)

	params['W' + name] = W
	params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
	res = np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

	return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
	"""
	Do a forward pass

	Keyword arguments:
		X -- input vector [Examples x D]
		params -- a dictionary containing parameters
		name -- name of the layer
		activation -- the activation function (default is sigmoid)
	"""
	pre, post_act = None, None
	# get the layer parameters
	W = params['W' + name]
	b = params['b' + name]

	# your code here
	# ﬁrst hidden layer pre-activation a(1)(x)
	pre = np.dot(X,W) + b
	# post-activation values of the ﬁrst hidden layer h(1)(x)
	post_act = activation(pre)

	# store the pre-activation and post-activation values
	# these will be important in backprop
	params['cache_' + name] = (X, pre, post_act)

	return post_act



# Q 2.2.2
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
	c=- np.max(x, axis = -1)
	Ex = np.exp(x.T +c)
	result = (Ex / sum(Ex))
	res=np.transpose(result)

	return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
	e = 1e-12
	probs = np.clip(probs, e, (1.0 - e))
	loss = np.sum(y*np.log(probs+1e-9))
	loss=-loss/probs.shape[0]

	diff = np.argmax(probs, axis=1) - np.argmax(y, axis=1)
    # if diff is zero ,it is correct
	correct = np.count_nonzero(diff==0)
	acc=correct/ diff.shape[0]
	return loss, acc

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):

    res = post_act*(1.0-post_act)

    return res


def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
	"""
	Do a backwards pass
	Keyword arguments:
	delta -- errors to backprop
	params -- a dictionary containing parameters
	name -- name of the layer
	activation_deriv -- the derivative of the activation_func
	"""
	grad_X, grad_W, grad_b = None, None, None
	# everything you may need for this layer
	W = params['W' + name]
	b = params['b' + name]
	X, pre, post_act = params['cache_' + name]
	# your code here
	# do the derivative through activation first
	# then compute the derivative W,b, and X
	if name == 'output':
		grad_W = np.dot(X.T,delta)
		grad_b = np.mean(delta, axis=0)
		grad_X = np.dot(delta,W.T)
	else:
		delta_hidden = activation_deriv(post_act)*delta # hidden error
		grad_W = np.dot(X.T,delta_hidden)
		grad_b = np.mean(delta_hidden, axis=0)
		grad_X = np.dot(delta_hidden,W.T)
	# store the gradients
	params['grad_W' + name] = grad_W
	params['grad_b' + name] = grad_b
	return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
	n_batches = round(x.shape[0] / batch_size) # number of batches (length/batch_size)
	batches_list =[]
	pos = np.arange(x.shape[0])
	pos_re=pos.reshape((x.shape[0],1))

	shuffle = np.hstack((x,y,pos_re)) # shuffle
	np.random.shuffle(shuffle)
	x_sf = shuffle[:,:x.shape[1]]
	y_sf = shuffle[:,x.shape[1]:-1]
	b_pos = shuffle[:,-1]
	for i in range(n_batches):
		x_b=x_sf[i*batch_size:(i+1)*batch_size,:]
		y_b=y_sf[i*batch_size:(i+1)*batch_size,:]
		batches_list.append((x_b,y_b))

	# batches=np.asarray(batches_list)
	return batches_list,b_pos
