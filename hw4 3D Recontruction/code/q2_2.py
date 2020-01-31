import numpy as np
import matplotlib.pyplot as plt
import helper
import submission as sub
from numpy.linalg import det
#
# '''
# Q2.2: Seven Point Algorithm
#     Input:  pts1, Nx2 Matrix
#             pts2, Nx2 Matrix
#             M, a scalar parameter computed as max (imwidth, imheight)
#     Output: Farray, a list of estimated fundamental matrix.
# '''
# def sevenpoint(pts1, pts2, M):
#     pts1 = pts1/M
#     pts2 = pts2/M
#     x1 = pts1[:, 0]
#     y1 = pts1[:, 1]
#     x2 = pts2[:, 0]
#     y2 = pts2[:, 1]
#
#     # SVD solve AF=0
#     A = np.stack([x2*x1,  x2*y1,  x2,  y2*x1,  y2*y1,  y2,  x1,  y1,  np.ones(len(x1))], axis=1)
#     (U, S, V) = np.linalg.svd(A)
#     V = np.transpose(V)
#     F1 = np.reshape(V[:, len(V)-1], (3, 3))
#     F2 = np.reshape(V[:, len(V)-2], (3, 3))
#
#     # Det(al * F1 + (1 - a) * F2) = 0
#     # func = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
#     # roots=fsolve(func, 0)
#
#     # func = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
#     f0=det(F2)
#     f1=det(F1)
#     fn1=det(-1 * F1 + (1 - (-1)) * F2)
#     f2=det(2 * F1 - F2)
#     fn2=det(-2 * F1 + 3 * F2)
#     a0=f0                           # a0=fun(0)
#     a1=2*(f1-fn1)/3-(f2-fn2)/12     # a1=2(fun(1)−fun(−1))/3−(fun(2)−fun(−2))/12
#     a2=0.5*f1+0.5*fn1-f0            # a2=0.5fun(1)+0.5fun(−1)−f0
#     a3=f1-a0-a1-a2                  # a3=fun(0)-a0-a1-a2
#     # a0+a1x+a^2x2+a3x^3=fun(x)
#     roots=np.roots([a0,a1,a2,a3])
#     roots=np.real(roots)
#     print('roots=',roots)
#
#     T = np.array([[1/M, 0,   0],
#                  [0,   1/M, 0],
#                  [0,   0,   1]])
#     Farray=np.zeros([3,3,len(roots)])
#     for i in range(len(roots)):
#         F=roots[i] * F1 + (1 - roots[i]) * F2
#         # [U1, S1, V1] = np.linalg.svd(F)
#         # S1[2]=0
#         # F = np.dot(U1,np.diag(S1)).dot(V1)
#         F=np.dot(np.transpose(T), F).dot(T)
#         Farray[:,:,i]=F
#     np.savez('../data/q2_2.npz', F=F, M=M, pts1=pts1,pts2=pts2)
#     return Farray

if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    N = data['pts1'].shape[0]  # N=110
    M = 640
    pts1 = data['pts1']
    pts2 = data['pts2']
    Farray = sub.sevenpoint(pts1[7:14,:], pts2[7:14,:], M)
    print('F=',Farray[:,:,2])
    helper.displayEpipolarF(im1, im2, Farray[:,:,2])
