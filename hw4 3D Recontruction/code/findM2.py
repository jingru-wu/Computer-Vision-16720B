import numpy as np
import submission as sub
from helper import camera2
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

if __name__ == '__main__':
    #load data
    data = np.load('../data/some_corresp.npz')
    intrinsics = np.load('../data/intrinsics.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    M = 640
    N = len(pts1)
    F = sub.eightpoint(pts1, pts2, M)

    E = sub.essentialMatrix(F, K1, K2)
    # test q3.1
    # print('Essential Matrix',E)
    E = E / E[2,2]

# projection matrix
    M1 = np.eye(3,4)
    C1 = np.dot(K1, M1)
    M2s = camera2(E)
    # print(M2s.shape)
    # P_list=np.zeros([N,3,4])
    Err_list=np.zeros(4)
    for i in range(M2s.shape[2]):
        C2 = np.dot(K1,M2s[:, :, i])
        [P, Err_list[i]] = sub.triangulate(C1,pts1,C2,pts2)

        if(min(P[:,2]) > 0):
            M2=M2s[:,:,i]
            print('best M2=',M2)
            print('C2=',C2)
            # print('Error=',Err_list[i])
            # np.savez('../jingruwu/data/q3_3.npz', M2= M2, C2=C2, P=P)
        else:
            print('pass')


