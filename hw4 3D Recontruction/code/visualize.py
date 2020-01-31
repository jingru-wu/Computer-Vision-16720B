import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import submission as sub
'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == '__main__':
    # load im1, im2
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # get pts1 and pts2
    temple_Coords = np.load('../data/templeCoords.npz')
    x1 = temple_Coords['x1']
    y1 = temple_Coords['y1']
    pts1 = np.zeros([len(x1), 2])
    pts2 = np.zeros([len(x1), 2])
    pts1[:, 0] = x1.flatten()
    pts1[:, 1] = y1.flatten()
    q2_1=np.load('../jingruwu/data/q2_1.npz')
    F=q2_1['F']
    for i in range(len(x1)):
        [pts2[i,0], pts2[i,1]] = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
# get C1 C2
    intrinsics = np.load('../jingruwu/data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    M1 = np.eye(3,4)
    C1 = np.dot(K1, M1)
    findC2=np.load('../jingruwu/data/q3_3.npz')
    C2 = findC2['C2']
    M2= findC2['M2']

    #save the matrix F, matrices M1, M2, C1, C2 which you used to generate the screenshots to the ﬁle q4 2.npz.
    # np.savez('../jingruwu/data/q4_2.npz',F=F,M1=M1, M2=M2, C1=C1, C2=C2)
    [P_3d,err]=sub.triangulate(C1,pts1,C2,pts2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(P_3d.shape)
    # plot 3d scatter
    ax.scatter(P_3d[:,0], P_3d[:,1], P_3d[:,2],s=2)
    plt.show()
