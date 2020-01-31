import numpy as np
import matplotlib.pyplot as plt

import scipy.signal
import cv2
import submission as sub
from helper import epipolarMatchGUI
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
'''

if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    pts1 = data['pts1']
    pts2 = data['pts2']

    N = data['pts1'].shape[0]  # N=110
    M = 640
    P1=pts1
    P2=np.zeros(P1.shape)
    F = sub.eightpoint(pts1, pts2, M)
    for i in range(P1.shape[0]):
        [P2[i,0],P2[i,1]]=sub.epipolarCorrespondence(im1, im2, F, P1[i,0], P1[i,1])
    # np.savez('../jingruwu/data/q4_1.npz',P1=P1,P2=P2)
    epipolarMatchGUI(im1, im2, F)


