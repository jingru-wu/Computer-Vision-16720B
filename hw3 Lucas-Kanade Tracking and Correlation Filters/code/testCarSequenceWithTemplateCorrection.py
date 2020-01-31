import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanade
import cv2
# write your script here, we recommend the above libraries for making your animation
#  Displaying at frames 1,100,200,300,400
if __name__ == '__main__':
    CarSquence=np.load('../data/carseq.npy')
    x1=59
    x2=145
    y1=116
    y2=151
    number_image = CarSquence.shape[2]
    rect=np.asarray([x1,y1,x2,y2])
    rects=np.zeros([number_image,4])
    rects[0]=rect
    for i in range(number_image-1):
        It = CarSquence[:,:,i]
        It1 = CarSquence[:,:,i+1]
        p = LucasKanade.LucasKanade(It, It1, rect)
        x1 = x1+p[0]
        x2 = x2+p[0]
        y1 = y1+p[1]
        y2 = y2+p[1]
        print('image number',i)
        rect=np.array([x1,y1,x2,y2])
        rects[i+1,:]=rect
        pt_0=rects[i+1,:]-rects[0,:]
        pt_0=np.array([pt_0[0],pt_0[1]])
        diff_p=LucasKanade.LucasKanade(CarSquence[:,:,0], It1, rects[0,:],pt_0)
        tmp =  diff_p- pt_0
        diff = tmp - p
        threshold=3
        # norm(delta)
        if np.linalg.norm(diff) < threshold:
            p=tmp
            x1 = x1+p[0]
            x2 = x2+p[0]
            y1 = y1+p[1]
            y2 = y2+p[1]
            rect=np.array([x1,y1,x2,y2])
            rects[i+1,:]=rect
        else:
            p=p
    rects2=rects
    np.save('../data/carseqrects-wcrt.npy', rects2)

