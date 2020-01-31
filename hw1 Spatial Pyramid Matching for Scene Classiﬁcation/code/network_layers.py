import numpy as np
import scipy.ndimage
import os,time


import imageio
import skimage.transform
import skimage.measure
import torchvision
from torchvision import transforms
import torchvision
import util
#Q3.1
def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	layer_num=len(vgg16_weights)
	for i in range(0,layer_num):
		layer_name=vgg16_weights[i][0]
		if layer_name=='conv2d':
			weights=vgg16_weights[i][1]
			bias=vgg16_weights[i][2]
			x = multichannel_conv2d(x,weights,bias)
		if layer_name=='relu':
			x = relu(x)
			x = np.asarray(x)
		if layer_name=='maxpool2d':
			size=vgg16_weights[i][1]
			x = max_pool2d(x,size)
		if layer_name=='linear':
			weights=vgg16_weights[i][1]
			bias=vgg16_weights[i][2]
			x=np.ndarray.flatten(x)
			x = linear(x,weights,bias)
#		print(i,layer_name,'Ouput Shape:',x.shape)  #test vagg network status

	return x
	pass

#Q3.1
def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	input_dim=x.shape[2]
	filters_num=weight.shape[0]
	reshape_size=x.shape[0]
	# features=[]
	filter_result=[]
	for j in range(0,filters_num):
		channels_result=np.zeros((reshape_size,reshape_size))
		for k in range(0,input_dim):
			x_split=x[:,:,k].reshape(reshape_size,reshape_size)
			conv_result=scipy.ndimage.convolve(x_split,weight[j,k,:,:],mode='nearest')
			channels_result=channels_result+conv_result
		filter_result.append(channels_result+bias[j])
	filter_result=np.asarray(filter_result)
	#convert shape(out_dim,h,w) into (h,w,outdim)
	features = np.transpose(filter_result,(1,2,0))
	return features

#Q3.1
def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)
	pass
#Q3.1
def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)  figure_map
	* size: pooling receptive fields          sqaure: size*size

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	input_dim=x.shape[2]
	h=x.shape[0]/size
	w=x.shape[1]/size
	output_dim=input_dim
#	pool_out = np.zeros((h,w,output_dim))
	pool_out=[]
	for i in range(input_dim):
		pool_out.append(skimage.measure.block_reduce(x[:,:,i],(size,size), np.max))
	pool_out=np.asarray(pool_out)

	#convert shape:(output_dim,H/size,W/size) into shape:(H/size,W/size,output_dim)
	pool_out=np.transpose(pool_out,(1,2,0))
	return pool_out
pass
#Q3.1
def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: np.ndarray of shape (input_dim)
	* weight: np.ndarray of shape (output_dim,input_dim)
	* bias: np.ndarray of shape (output_dim)

	[output]
	* y: np.ndarray of shape (output_dim)
	'''
	y=np.matmul(W,x)
	y=y+b
	return y
pass
