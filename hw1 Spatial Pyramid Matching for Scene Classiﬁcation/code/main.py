import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import visual_words
import visual_recog
import deep_recog

import skimage.filters
import skimage.io
import imageio
import random # creat 3 random word maps
from scipy import ndimage
import multiprocessing

if __name__ == '__main__':

    num_cores = util.get_num_CPU()
    print(num_cores)


    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

# #Q1.1
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

# Q1.2
    visual_words.compute_dictionary(num_workers=num_cores)

# Q1.3 visualize word map
    dictionary = np.load('../code/dictionary.npy',allow_pickle=True)
    train_data = np.load('../data/train_data.npz',allow_pickle=True)
    train_name = train_data['image_names']
    train_data= np.random.permutation(train_name)[:3] # load 3 random images for visulize wordmap
    for image in train_data:
        image = imageio.imread('../data/'+ image[0])
        wordmap = visual_words.get_visual_words(image, dictionary)
        f, axes = plt.subplots(1, 2)
        f.set_size_inches(8, 8)
        axes[0].imshow(image)
        axes[0].set_title('original')
        axes[1].imshow(wordmap)
        axes[1].set_title('wordmap')
        plt.show()

# Q2.1  plot feature of one word map
    dictionary = np.load('../code/dictionary.npy',allow_pickle=True)
    wordmap = visual_words.get_visual_words(image, dictionary)
    dict_size=len(dictionary)
    hist_all = visual_recog.get_feature_from_wordmap(wordmap,dict_size)
    plt.hist(hist_all,dict_size)
    plt.show()

#Q2.4
    visual_recog.build_recognition_system(num_cores)

##Q2.5
    visual_recog.evaluate_recognition_system(num_cores)

##Q3
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()

#Q3.2
    deep_recog.build_recognition_system(vgg16,num_workers=num_cores)

#Q3.2
    conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())
