# write your script here, we recommend the above libraries for making your animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SubtractDominantMotion import SubtractDominantMotion
import skimage.transform
import scipy.ndimage
import cv2
import LucasKanadeAffine
# write your script here, we recommend the above libraries for making your animation
#  Displaying at frames 1,100,200,300,400
if __name__ == '__main__':
    Aerial=np.load('../data/aerialseq.npy')
    number_image = Aerial.shape[2]
    rects=np.zeros([number_image,4])
    for i in range(number_image-1):
        It = Aerial[:,:,i]
        It1 = Aerial[:,:,i+1]
        mask=SubtractDominantMotion(It,It1)
        print('frame:',i)
        I_tack_P=It1.copy()
        I_tack_P[mask!=0]=1
        I_tack=np.dstack((It1,It,I_tack_P))
        if i==30 or i==60 or i==90 or i==120:
        # if i%5==0:
            plt.imshow(I_tack),plt.title('It')
            plt.title('frame: %i'%i)
            plt.show()
