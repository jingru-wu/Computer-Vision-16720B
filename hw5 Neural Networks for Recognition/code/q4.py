import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# Q 4.2
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    boxes_list = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    sigma = 0.1
    random = skimage.util.random_noise(image, var=sigma**2)
    image = skimage.restoration.denoise_tv_chambolle(random, weight=0.1, multichannel=True)

    #image = skimage.restoration.denoise_bilateral(image)
    #greyscale
    # apply threshold
    im_g = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(im_g)
    bw = skimage.morphology.closing(im_g < thresh,skimage.morphology.square(6))

    #bw = skimage.morphology.erosion(bw)
    bw = skimage.morphology.dilation(bw)
    # remove artifacts connected to image border
    clear = skimage.segmentation.clear_border(bw)
    #bw = cleared
    # label image regions
    label_region = skimage.measure.label(clear)

    bw = np.invert(bw).astype(int)
    #boxes_list = skimage.measure.regionprops(label_image)
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in skimage.measure.regionprops(label_region):
    # take regions with large enough areas
        if region.area >= 300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            small_box = [minr, minc, maxr, maxc ]
            boxes_list.append(small_box)
    boxes_list = np.matrix(boxes_list)
    return boxes_list, bw
