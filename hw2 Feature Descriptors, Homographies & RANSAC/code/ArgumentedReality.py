import numpy as np
import planarH
import skimage.io
import matplotlib.pyplot as plt

import scipy
def compute_extrinsics(K, H):
    '''
    Input:
    K ：3*3：intrinsic parameters
    H ： homography

    output:
    R: 3*3: rotation
    '''
    ######################################
    H1=np.dot(np.linalg.inv(K),H)
    [U,S,V]= np.linalg.svd(H1[:,:2])
    S=np.array([[1,0],[0,1],[0,0]])
    R12=np.dot(np.dot(U,S),V)
    R12=np.array(R12)
    R3=np.cross(R12[:,0],R12[:,1])
    R=np.vstack((R12[:,0],R12[:,1],R3))
    R=np.transpose(R)
    det=np.linalg.det(R)
    R[:,2]=R[:,2]*(det/abs(det))
    ratio=np.mean(H1[:,:2]/R[:,:2])
    t=H1[:,2]/ratio
    return R,t



def project_extrinsics(K, W, R, t):



    dp=np.matrix([[5],[9.65],[6.858/2]])
    W1=W+dp


    X1=np.dot(R,W1)+t
    X1=np.dot(K,X1)
    X1=X1/X1[2,:]

    X2=np.dot(R,W)+t
    X2=np.dot(K,X2)
    X2=X2/X2[2,:]

    plt.plot(np.squeeze(np.asarray(X1[0,:])),np.squeeze(np.asarray(X1[1,:])),'y.',markersize=1)
    plt.show()
    return X1




if __name__ == '__main__':
    W =np.array([[0.0,18.2,18.2,0.0],
                 [0.0,0.0,26.0,26.0],
                 [0.0,0.0,0.0,0.0]])
    X =np.array([[483,1704,2175,67],
                 [810,781,2217,2286]])
    K =np.array([[3043.72, 0.0,     1196.00],
                 [0.0,     3043.72, 1604.00],
                 [0.0,     0.0,     1.0]])
    H=planarH.computeH(X,W[:2,:])
    R,t=compute_extrinsics(K,H)
    print('R:\n',R)
    print('t:\n',t)
    W = np.loadtxt('..\data\sphere.txt')
    im=skimage.io.imread('..\data\prince_book.jpeg')
    plt.imshow(im)
    X=project_extrinsics(K, W, R, t)

