import numpy as np
import matplotlib.pyplot as plt
import helper
import submission as sub
# #Q2.1: Eight Point Algorithm
# '''
#     Input:  pts1, Nx2 Matrix
#             pts2, Nx2 Matrix
#             M, a scalar parameter computed as max (imwidth, imheight)
#     Output: F, the fundamental matrix 3*3
# '''
# def eightpoint(pts1, pts2, M):
#     pts1 =pts1/M
#     pts2 =pts2/M
#     x1 = pts1[:,0]
#     y1 = pts1[:,1]
#     x2 = pts2[:,0]
#     y2 = pts2[:,1]
#
#     # SVD solve AF=0
#     A = np.stack([x2*x1,  x2*y1,  x2,  y2*x1,  y2*y1,  y2,  x1,  y1,  np.ones(len(x1))],axis=1)
#     (U, S, V) = np.linalg.svd(A)
#     V = np.transpose(V)
#     F = np.reshape(V[:,len(V)-1],(3,3))
#
#     # enforce the singularity condition of the F before unscaling  Set rank=2
#     [U1, S1, V1] = np.linalg.svd(F)
#     S1[2]=0
#     F = np.dot(U1,np.diag(S1))
#     F=np.dot(F, V1)
#     # refine F before unscaling F
#     F = helper.refineF(F, pts1, pts2)
#
#     # unscaling F
#     T = np.array([[1/M, 0,   0],
#                 [0,   1/M, 0],
#                 [0,   0,   1]])
#     F = np.dot(np.transpose(T), F).dot(T)
#     # np.savez('../data/q2_1.npz', F=F, M=M)
#     return F

if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]  # N=110
    M = 640
    pts1 = data['pts1']
    pts2 = data['pts2']
    F8 = sub.eightpoint(pts1, pts2, M)
    print(F8)
    helper.displayEpipolarF(im1, im2, F8)


