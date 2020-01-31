import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
from scipy.ndimage import gaussian_filter, gaussian_laplace
import skimage.io
def image_preshape(image):
    if image.shape[2]==2:
        image = np.dstack((image,image,image))
    if image.shape[2]==4:
        image = np.delete(image,-1,axis=2)
    image=skimage.color.rgb2lab(image)
    return image


def filter1(im, scale):
    return gaussian_filter(im, sigma=scale, output=np.float64, mode='nearest')
def filter2(im, scale):
    return gaussian_laplace(im, sigma=scale, output=np.float64, mode='nearest')
def filter3(im, scale): # gaussian deviation in x direction
    return gaussian_filter(im, sigma=scale, order=[1,0], output=np.float64, mode='nearest')
def filter4(im, scale):
    return gaussian_filter(im, sigma=scale, order=[0,1], output=np.float64, mode='nearest')

def extract_filter_responses(image):
    '''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # ----- TODO -----
    image=image_preshape(image)
    result=[]
    for scale in [1.0,2.0,4.0,8.0,8.0*np.sqrt(2)]:
        for f in[filter1,filter2,filter3,filter4]:
            for im in [image[:,:,0],image[:,:,1],image[:,:,2]]:
                filtered=f(im,scale)
                result.append(filtered)
    filter_responses=np.dstack(result)
    return filter_responses


def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.
    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * reshaped_wordmap: numpy.ndarray of shape (H,W)
    '''
    # ----- TODO -----
    filter_response = extract_filter_responses(image)
    filter_response=filter_response.reshape(-1, 60)
    distance=scipy.spatial.distance.cdist(filter_response,dictionary,metric='euclidean')
    best_fit = np.argmin(distance, axis=1)
    reshaped_wordmap=best_fit.reshape(image.shape[0], image.shape[1])
    return reshaped_wordmap

def compute_dictionary_one_image(i,alpha,args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time
    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    #i,alpha,image_path = args
    # ----- TODO -----

    load_path=os.path.join('../data/', args[i][0])
    save_path = '../temp_files/'
    image = skimage.io.imread(load_path)
    filter_responses = extract_filter_responses(image)
    reshaped_response=filter_responses.reshape(-1, filter_responses.shape[2]) #reshape to
    shuffle = np.random.permutation(reshaped_response)
    np.save('../temp_files/{}.npy'.format(i), shuffle[:alpha])
    print('Finished image dictionary:', i)


def compute_dictionary(num_workers):
    '''
    Creates the dictionary of visual words by clustering using k-means.
    [input]
    * num_workers: number of workers to process in parallel
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    # ----- TODO -----
    K = 200
    train_file = np.load("../data/train_data.npz",allow_pickle=True)
    train_data = train_file['image_names']
    length=len(train_data)
    args = [[i, 250, train_data] for i in range(length)]
        # p = Pool(processes=num_workers)
        # p.map(func, args)
        # p.close()
        # p.join()
        # p.terminate()
        # p.map_async
    with multiprocessing.Pool(processes=num_workers) as p:
        p.starmap(compute_dictionary_one_image, args)
    features = []
    for file in os.listdir('../temp_files/'):
        temp = np.load('../temp_files/' + file)
        features.append(temp)
        features = np.asarray(features)
    filter_responses = features.reshape(-1, 60)
    print('K-means Clustering Started')
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    print('K-means Clustering Finished')
    # np.save('../code/dictionary.npy', dictionary)
    print('dictionary has benn saved')
    return dictionary

