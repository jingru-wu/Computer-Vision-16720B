import numpy as np
import matplotlib.pyplot as plt
import helper
import submission as sub
from mpl_toolkits.mplot3d import Axes3D
if __name__ == '__main__':
    data=np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    pt1=data['pts1']
    pt2=data['pts2']

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    M=650
    [F, inliers]=sub.ransacF(pt1,pt2,M)
    pts1_inliers=pt1[inliers,:]
    pts2_inliers=pt2[inliers,:]
    F = sub.eightpoint(pts1_inliers, pts2_inliers, M)

    # F=sub.eightpoint(pt1,pt2,M)
    # pts1_inliers=pt1
    # pts2_inliers=pt2

    E = sub.essentialMatrix(F, K1, K2)
    E = E / E[2,2]
    M1 = np.eye(3,4)
    C1 = np.dot(K1, M1)
    M2s = helper.camera2(E)
    M2_init=np.zeros([3,4])
    C2 = np.zeros([3,4])
    for i in range(M2s.shape[2]):
        C2 = np.dot(K1,M2s[:, :, i])
        [P, err] = sub.triangulate(C1,pts1_inliers,C2,pts2_inliers)
        if(min(P[:,2]) > 0):
            M2_init=M2s[:,:,i]
            print('initial M2',M2_init)
            P_init=P
        else:
            print('pass')


    [M2,P_optimized]=sub.bundleAdjustment(K1, M1, pts1_inliers, K2, M2_init, pts2_inliers, P_init)
    print('optimized M2',M2)

    P_optimized=P_optimized.reshape(-1,3)
    print('optimized P shape:',P_optimized.shape)
    # plot 3d scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_init[:,0], P_init[:,1], P_init[:,2],s=2,c='b')
    ax.scatter(P_optimized[:,0], P_optimized[:,1], P_optimized[:,2],s=2,c='r')
    plt.show()
