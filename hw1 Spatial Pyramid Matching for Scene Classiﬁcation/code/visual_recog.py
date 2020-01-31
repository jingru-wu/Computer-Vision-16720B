import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing

import matplotlib.pyplot as plt
import random

##Q2.4
def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	#variables initialise
	features=[]
	train_data = np.load("../data/train_data.npz",allow_pickle=True)
	dictionary = np.load("../code/dictionary.npy",allow_pickle=True)
	train_name,labels = train_data['image_names'],train_data['labels']

	SPM_layer_num = 3
	K = 200
	T = len(train_name)

	for i in range(T):
		features.append(get_image_feature(train_name[i],dictionary,3,K))
		print('Get feature of train data:',i)
	features = np.array(features)
	#
	# # mutiprocess
	# arg =[[train_name[i],dictionary,SPM_layer_num,K,i] for i in range(T)]
	# with multiprocessing.Pool(num_workers) as p:
	#  	features=p.starmap(get_image_feature,arg)
	#  	features.append(features)
	# features = np.array(features)

	print('Finished all features of train data')
	np.savez('../code/trained_system.npz', features=features, labels=labels,dictionary=dictionary,SPM_layer_num=SPM_layer_num)
	pass
##Q2.5
def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	# # ----- TODO -----
	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("../code/trained_system.npz")
	dictionary=np.load("../code/dictionary.npy")
	test_data, excepted_labels = test_data['image_names'], test_data['labels']
	train_features,train_labels = trained_system['features'],trained_system['labels']

	recognized_labels = np.zeros((len(train_labels)))
	i=0
	for test_image in test_data:
		print('recognize:',i)
		word_hist = get_image_feature(test_image, dictionary,3,200)
		sim = distance_to_set(word_hist, train_features)
		similarity_index=np.argmax(sim)
		recognized_labels[i]=train_labels[similarity_index]
		i=i+1
	recognized_labels = np.asarray(recognized_labels)

	Conf = np.zeros((8,8))
	lenth=len(excepted_labels)
	error_list=[]
	for x in range(0,lenth):
		i=int(excepted_labels[x])
		j=int(recognized_labels[x])
		Conf[i,j] = Conf[i,j]+1
		if (i!=j):
			error_list.append(test_data[x])
	error_list=np.asarray(error_list)
	np.save('error_list.npy',error_list)
	correct_num=np.trace(Conf)
	all_num=len(excepted_labels)
	accuracy = correct_num/all_num
	print('Accuracy is', accuracy*100,'%')
	print(Conf)
##Q2.6 plot 3 error image and wordmap from error_list
	for i in range(5):
		error_image=imageio.imread('../data/'+error_list[i][0])
		error_wordmap=visual_words.get_visual_words(error_image,dictionary)
		f, axes = plt.subplots(1,2)
		axes[0].imshow(error_image)
		axes[0].set_title('error image')
		axes[1].imshow(error_wordmap)
		axes[1].set_title('error wordmap')
		plt.show()

	return Conf,accuracy
pass

def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	# ----- TODO -----
	# layer_num=3, K=200
	# load image
	image_path = os.path.join('../data/', file_path[0])
	image = imageio.imread(image_path)
	# extract word map from the image,
	wordmap = visual_words.get_visual_words(image, dictionary)
	# compute SPM feature
	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, len(dictionary))
	# return the feature
	return feature
	pass


##Q2.3
def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
# ----- TODO -----
	sim = np.minimum(word_hist, histograms).sum(axis=1)
	return sim

	pass

##Q2.1
def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	# ----- TODO -----
	hist,edges= np.histogram(wordmap,dict_size)
#	print('cell histogram finished')
	return hist


##Q2.2
def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''

	# ----- TODO -----
	Layer = layer_num - 1
	cell_h = int(wordmap.shape[0]/2**Layer)
	cell_w = int(wordmap.shape[1]/2**Layer)
	layer_0 = []
	layer_1 = []
	layer_2 = []
	for i in range(2**Layer):
		for j in range(2**Layer):
			seperated_hist = get_feature_from_wordmap(wordmap[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w],dict_size)
			layer_2.append(seperated_hist)
	layer_2 = np.asarray(layer_2)
	layer_1.append(layer_2[0]+layer_2[1]+layer_2[4]+layer_2[5])
	layer_1.append(layer_2[2]+layer_2[3]+layer_2[6]+layer_2[7])
	layer_1.append(layer_2[8]+layer_2[9]+layer_2[12]+layer_2[13])
	layer_1.append(layer_2[10]+layer_2[11]+layer_2[14]+layer_2[15])
	layer_1 = np.asarray(layer_1)
	layer_0 = layer_1[0]+layer_1[1]+layer_1[2]+layer_1[3]
	hist_all = np.concatenate((0.25*layer_0, 0.25*layer_1.flatten(), 0.5*layer_2.flatten()))#size=(4200=16+4+1)*200
	num_features = hist_all.sum()
	hist_all = hist_all/num_features   # normalized
	sum=hist_all.sum()                 # test normalization
#	print('whole histogram of this wordmap is finished')
	return hist_all

	pass






	

