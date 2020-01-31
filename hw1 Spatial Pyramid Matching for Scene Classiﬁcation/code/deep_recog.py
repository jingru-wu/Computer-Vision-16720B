import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util

import torch
import network_layers
from scipy.spatial import distance
#Q3.2
def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	# ----- TODO -----
	num_workers=util.get_num_CPU()
	vgg16_weights=util.get_VGG16_weights()
	train_data = np.load("../data/train_data.npz")
	train_name,labels = train_data['image_names'],train_data['labels']

	lenth=len(train_name)
	vgg16=torchvision.models.vgg16(pretrained=True)

	features=[]
	for i in range(0,lenth):
		image_path = os.path.join('../data/',train_name[i][0])
		args=[i,image_path,vgg16]
		single_feature=get_image_feature(args)
		features.append(single_feature)
	features=np.asarray(features)
	#np.savez('../code/trained_system_deep.npz', features=features, labels=labels)
	print('saved train_system_deep.npz')
	return features,labels



	# # # this part is to bulid a trained system by using functions writen by myself
	# lenth=int(len(train_name)/36)
	#
	# #-- mutiprocess get deep features from my functions
	# arg =[[train_name[i],labels[i],vgg16_weights,i] for i in range(0,lenth)]
	# with multiprocessing.Pool(8) as p:
	# 	p.starmap(process,arg)
	# print('Finished all features of train dataset')
	# # # convert single feature and label to deep system
	# features=[]
	# labels=[]
	# for i in range(0,lenth):
	# 	single_feature=np.load('../deep_features/{}.npy'.format(i))
	# 	single_label=single_feature[1000]
	# 	single_feature=single_feature[:1000]
	# 	features.append(single_feature)
	# 	labels.append(single_label)
	# features=np.array(features)
	# labels=np.array(labels)
	# np.savez('../code/trained_system_deep.npz', features=features, labels=labels)
	# print('saved train_system_deep.npz')

pass

#Q3.2
def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	# ----- TODO -----
	test_data = np.load("../data/test_data.npz")
	test_data, predicted_labels = test_data['image_names'], test_data['labels']
	trained_system = np.load("../code/trained_system_deep.npz")
	train_features,train_labels = trained_system['features'],trained_system['labels']
	vgg16_weights=util.get_VGG16_weights()


	labels = []
	i=0
	for i in range(0,len(test_data)):
		print('vgg recognization:',i)
		image_path = os.path.join('../data/',test_data[i][0])
		args=[i,image_path,vgg16]
		feature = get_image_feature1(args)
		dist = distance_to_set(feature,train_features)
		distance_index=np.argmin(dist)
		recognized_deep_labels=train_labels[distance_index]
		labels.append(recognized_deep_labels)
	recognized_deep_labels = np.array(labels)

	conf = np.zeros((8,8))
	all_num=len(predicted_labels)
	for x in range(0,all_num):
		i=predicted_labels[x]
		j=labels[x]
		conf[i,j] = conf[i,j]+1
	correct_num=np.trace(conf)
	accuracy = correct_num/all_num
	print('Accuracy is', accuracy*100,'%')
	print(conf)
	return conf,accuracy
	pass

def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (1,3,H,W)
	'''
	# ----- TODO -----
	mean=[0.485,0.456,0.406]
	std=[0.229,0.224,0.225]
	# image_processed=np.zeros((1,3,224,224))
	image=skimage.transform.resize(image,(224,224,3))
	image = np.transpose(image,(2,0,1))
	image = torch.from_numpy(image).type(torch.FloatTensor)
	normalize=torchvision.transforms.Normalize(mean,std)
	image=normalize(image)
	image_processed = image.unsqueeze(0)
	# image_processed=torch.from_numpy(image_processed)
	# image_processed[0,:,:,:]=image
	# del image
	return image_processed
	pass

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.

 	[saved]
	* x: evaluated deep feature
	'''

	i,image_path,vgg16 = args

	# ----- TODO -----
	vgg16_weights=util.get_VGG16_weights()
	path = os.path.join('../data/',image_path)
	image = imageio.imread(path)
	image=preprocess_image(image)

	x = vgg16.features(image.float()).detach().numpy()  #x is the conv output of first 30 laryers
	## go through the rest layers of Vgg16
	layer_num=len(vgg16_weights)
	for j in range(31,layer_num):
			layer_name=vgg16_weights[j][0]
			if layer_name=='relu':
				x = network_layers.relu(x)
				x = np.asarray(x)
			if layer_name=='linear':
				weights=vgg16_weights[j][1]
				bias=vgg16_weights[j][2]
				x=np.ndarray.flatten(x)
				x = network_layers.linear(x,weights,bias)
	print('get deep feature:',i)
	del image
	return x
	pass

def get_image_feature1(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.

 	[saved]
	* x: evaluated deep feature
	'''

	i,image_path,vgg16 = args

	# ----- TODO -----
	vgg16_weights=util.get_VGG16_weights()
	path = os.path.join('../data/',image_path)
	image = imageio.imread(path)
	image=preprocess_image(image)

	x = vgg16.features(image.double()).detach().numpy()  #x is the conv output of first 30 laryers
	## go through the rest layers of Vgg16
	layer_num=len(vgg16_weights)
	for j in range(31,layer_num):
			layer_name=vgg16_weights[j][0]
			if layer_name=='relu':
				x = network_layers.relu(x)
				x = np.asarray(x)
			if layer_name=='linear':
				weights=vgg16_weights[j][1]
				bias=vgg16_weights[j][2]
				x=np.ndarray.flatten(x)
				x = network_layers.linear(x,weights,bias)
	print('get deep feature:',i)
	del image
	return x
	pass






def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	# ----- TODO -----
	N=train_features.shape[0]
	dist=np.zeros(N)
	for n in range(0,N):
		dist[n]= distance.euclidean(feature,train_features[n])
	return dist
	pass

## to call my won extract feature function
def process(train_files,label,vgg16_weights,i):

	feature=np.zeros((1001))
	path = os.path.join('../data/',train_files[0])
	image = imageio.imread(path).astype('double')
	image=skimage.transform.resize(image,(224,224,3))

	feature[:1000]=network_layers.extract_deep_feature(image,vgg16_weights)
	feature[1000]=label
	#add label to the end of the feature
	feature_label=np.asarray(feature)
	np.save('../deep_features/{}.npy'.format(i), feature_label)
	print('get deep features:',i,'shape:',feature_label.shape)
	return feature
pass


if __name__ == '__main__':
	vgg16 = torchvision.models.vgg16(pretrained=True).double()
	vgg16.eval()
#	build_recognition_system(vgg16,8)
	num_workers=util.get_num_CPU()
	evaluate_recognition_system(vgg16,num_workers)
	print('all finished')
