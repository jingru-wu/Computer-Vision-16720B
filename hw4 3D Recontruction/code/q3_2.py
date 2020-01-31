import numpy as np
import submission
'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    P = np.zeros([N, 4])
    err=np.zeros(N)# matrix of 3D coordinates
    for i in range(N):
    # generate AP = 0  A: 4*4
        A = np.vstack([pts1[i, 0] * C1[2, :] - C1[0, :],
                       pts1[i, 1] * C1[2, :] - C1[1, :],
                       pts2[i, 0] * C2[2, :] - C2[0, :],
                       pts2[i, 1] * C2[2, :] - C2[1, :]])
        (U, S, V) = np.linalg.svd(A)
        V = np.transpose(V)
        V=V[:,3]
        V= V/V[3]
        P[i, :] = V   # N * 3
    # calculate Error
    p_1 = np.dot(C1, np.transpose(P)) # 3*N
    temp_1=p_1/np.array([p_1[2,:]])
    p_2 = np.dot(C2, np.transpose(P))
    temp_2=p_2/np.array([p_2[2,:]])
    p_1=temp_1[:2,:].transpose()
    p_2=temp_2[:2,:].transpose()
    err = sum(np.sqrt(np.transpose((pts1 - p_1)**2)) + np.sqrt(np.transpose((pts2 - p_2)**2)))
    err = sum(err)
    #     mm1=C1.dot(V)
    #     mm1=mm1/mm1[-1]
    #     mm2=C2.dot(V)
    #     mm2=mm2/mm2[-1]
    #     err[i]=np.linalg.norm(pts1[i,:]-mm1[0:-1])+ np.linalg.norm(pts2[i,:]-mm2[0:-1])
    err=np.sum(err)
    P=P[:, 0:3]
    return P, err

