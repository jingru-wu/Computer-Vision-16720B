import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
#-----------------To do------------------
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],[0,0,1]])

# initialize parameters
    threshold=0.1
    tre_p=10
    p0=np.array([0,0,0,0,0,0])
    gradient= np.asarray(np.gradient(It1))
    D_x=gradient[1]
    D_y=gradient[0]
    H,W=It1.shape
    X_range=np.arange(W)
    Y_range=np.arange(H)
    # X,Y=np.meshgrid(X_range,Y_range)
    # X=X.flatten()
    # Y=Y.flatten()
    Interation=0
    while tre_p>threshold:
        Interation=Interation+1
        print(Interation)
        M=np.array([[1+p0[0], p0[2],    p0[4] ],
                   [p0[1],    1+p0[3],  p0[5] ],
                   [0,        0,        1.    ]])
        # It1_w = scipy.ndimage.affine_transform(It1, M)
        Coodinate=np.asarray([[X_range],
                             [Y_range],
                             [np.ones(1,H*W)]])





        SDQ=np.stack([D_x*X,
                      D_y*X,
                      D_x*Y,
                      D_y*Y,
                      D_x,
                      D_y],axis=1)
        H = np.dot(np.transpose(SDQ),SDQ)   # Hessian matrix: 2*2
        # mask=It1_w==0
        # It_m=It.copy()
        # It_m[mask]=0
        Error=It-It1_w   #N*1
        B= SDQ.T.dot(Error.flatten())    # 6*1 = 6*N * N*1
        delta_p =np.dot(np.linalg.inv(H),B)  # delta p: 6*1[ px ; py]
        p0 = p0+delta_p
        tre_p = np.linalg.norm(delta_p)
        print('normal(delta_p)',tre_p)
    print('affine finished')
    return M

if __name__ == '__main__':
    Aerial=np.load('../data/aerialseq.npy')
    number_image = Aerial.shape[2]
    rects=np.zeros([number_image,4])
    for i in range(number_image-1):
        It = Aerial[:,:,i]
        It1 = Aerial[:,:,i+1]
#        m=SubtractDominantMotion(It,It1)
        print(i)
        W=LucasKanadeAffine(It,It1)
        print(W)
        It_w= cv2.warpAffine(It1,W[0:2,:],(It1.shape[1],It1.shape[0]))
        plt.subplot(1,3,1),plt.imshow(It),plt.title('It')
        plt.subplot(1,3,2),plt.imshow(It_w),plt.title('It_w')
        plt.subplot(1,3,3),plt.imshow(It1),plt.title('It1')
        plt.show()
