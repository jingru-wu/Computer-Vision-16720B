"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import cv2
import helper
import scipy
from numpy.linalg import det
import math

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    pts1 =pts1/M
    pts2 =pts2/M
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    # SVD solve AF=0
    A = np.stack([x2*x1,  x2*y1,  x2,  y2*x1,  y2*y1,  y2,  x1,  y1,  np.ones(len(x1))],axis=1)
    (U, S, V) = np.linalg.svd(A)
    V = np.transpose(V)
    F = np.reshape(V[:,len(V)-1],(3,3))

    # enforce the singularity condition of the F before unscaling  Set rank=2
    [U1, S1, V1] = np.linalg.svd(F)
    S1[2]=0
    F = np.dot(U1,np.diag(S1))
    F=np.dot(F, V1)
    # refine F before unscaling F
    F = helper.refineF(F, pts1, pts2)

    # unscaling F
    T = np.array([[1/M, 0,   0],
                [0,   1/M, 0],
                [0,   0,   1]])
    F=np.dot(np.transpose(T), F).dot(T)
    # np.savez('../jingruwu/data/q2_1.npz', F=F, M=M)
    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    pts1 = pts1/M
    pts2 = pts2/M
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    x2 = pts2[:, 0]
    y2 = pts2[:, 1]

    # SVD solve AF=0
    A = np.stack([x2*x1,  x2*y1,  x2,  y2*x1,  y2*y1,  y2,  x1,  y1,  np.ones(len(x1))], axis=1)
    (U, S, V) = np.linalg.svd(A)
    V = np.transpose(V)
    F1 = np.reshape(V[:, len(V)-1], (3, 3))
    F2 = np.reshape(V[:, len(V)-2], (3, 3))

    # Det(al * F1 + (1 - a) * F2) = 0
    # func = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    # roots=fsolve(func, 0)

    # func = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
    f0=det(F2)
    f1=det(F1)
    fn1=det(-1 * F1 + (1 - (-1)) * F2)
    f2=det(2 * F1 - F2)
    fn2=det(-2 * F1 + 3 * F2)
    a0=f0                           # a0=fun(0)
    a1=2*(f1-fn1)/3-(f2-fn2)/12     # a1=2(fun(1)−fun(−1))/3−(fun(2)−fun(−2))/12
    a2=0.5*f1+0.5*fn1-f0            # a2=0.5fun(1)+0.5fun(−1)−f0
    a3=f1-a0-a1-a2                  # a3=fun(0)-a0-a1-a2
    # a0+a1x+a^2x2+a3x^3=fun(x)
    roots=np.roots([a0,a1,a2,a3])
    roots=np.real(roots)
    # print('roots=',roots)

    T = np.array([[1/M, 0,   0],
                 [0,   1/M, 0],
                 [0,   0,   1]])
    Farray=np.zeros([3,3,len(roots)])
    for i in range(len(roots)):
        F=roots[i] * F1 + (1 - roots[i]) * F2
        # [U1, S1, V1] = np.linalg.svd(F)
        # S1[2]=0
        # F = np.dot(U1,np.diag(S1)).dot(V1)
        F=np.dot(np.transpose(T), F).dot(T)
        Farray[:,:,i]=F
    # np.savez('../jingruwu/data/q2_2.npz', F=F, M=M, pts1=pts1,pts2=pts2)
    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.dot(np.transpose(K2),F)
    E = np.dot(E,K1)
    return E

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
def epipolarCorrespondence(im1, im2, F, x1, y1):
    if len(im1.shape)==3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    W=im2.shape[0]
    H=im2.shape[1]

    p1=np.vstack([x1,y1,1])
    epipolar_line = np.dot(F,p1)
    a = epipolar_line[0]
    b = epipolar_line[1]
    c = epipolar_line[2]

    win_size=30
    hs=win_size/2  # half window size
    sigma = 5
    min_distance = 10e4  #set the initial value of distance to be infinity
    # set up filter
    # Gaussian_filter = scipy.signal.gaussian(win_size, sigma)
    patch1 = im1[int((y1 - hs)):int((y1 + hs)), int((x1 - hs)):int((x1 + hs))]
    # small patch in im1 centered at x1, y1
    errlist=[]
    # iterating along the epline to find matched x2, y2
    for y2 in range(int(y1-20),int(y1+20)):
        # compute x2 for each y2: a*x2 + b*y2 + c = 0
        x2 = np.round((-b * y2 - c) / a)
        if (x2 >= hs and x2<= H-hs and y2>=hs and y2<=W-hs):
            patch2 = im2[int(y2-hs):int(y2+hs),int(x2-hs):int(x2+hs)]
            # # patch in im2 centered at x2, y2
            # weighted_patch1 = np.dot(Gaussian_filter,patch1)
            # weighted_patch2 = np.dot(Gaussian_filter,patch2)
            # weighted_distance = abs(weighted_patch1-weighted_patch2)
            # err = np.sqrt(sum(weighted_distance**2))

            weighted_patch1 = patch1
            weighted_patch2 = patch2
            weighted_distance = abs(weighted_patch1-weighted_patch2)
            err = sum(np.sqrt(sum(weighted_distance**2)))

            errlist.append(err)
            if err < min_distance:
                min_distance = err
                x2_n = int(x2)
                y2_n = int(y2)
    errlist=np.asarray(errlist)
    # print(errlist)
    return x2_n,y2_n

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation

    num_iter=5000
    tol=8*10e-5
    #create a set of random index to match points
    randIdx=np.random.randint(0,pts1.shape[0]-1,size=num_iter*7,dtype=int)
    randIdx=randIdx.reshape(num_iter,7)
    max_inliers=0
    for i in range(num_iter):
        index=randIdx[i,:]
        P1=pts1[index,:]
        P2=pts2[index,:]
        Farray=sevenpoint(P1,P2,M)
        inliers_points=np.repeat(False,len(pts1))
        for j in range(Farray.shape[2]):

            num_inliers=0
            F=Farray[:,:,j]
            for m in range(len(pts1)):
                p2=np.hstack([pts2[m,0],pts2[m,1],1])
            # a = epipolar_line[0]
            # b = epipolar_line[1]
            # c = epipolar_line[2]
                p1=np.vstack([pts1[m,0],pts1[m,1],1])
                epipolar_line = np.dot(F,p1)
                diff= abs(np.dot(p2,epipolar_line))
                if diff<tol:
                    num_inliers=num_inliers+1
                    inliers_points[m]=True
            if num_inliers>max_inliers:
                max_inliers=num_inliers
                inliers=inliers_points
                F_best=F
        print('time:',i,'inliers',num_inliers)
    print('max num of inlier points',max_inliers)
    # pts1_inliers=pts1[inliers,:]
    # pts2_inliers=pts2[inliers,:]
    # F_best = eightpoint(pts1_inliers, pts2_inliers, M)
    return F_best,inliers



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta=np.linalg.norm(r)
    K=np.vstack(r/theta)
    sin=math.sin(theta)
    cos=math.cos(theta)
    Kx=np.array([[0.0,   -K[2], K[1]],
                 [K[2],  0.0,   -K[0]],
                 [-K[1], K[0],  0]])
    KK=np.dot(K,np.transpose(K))
    Icos=np.diag([cos,cos,cos])
    R=Icos+sin*Kx+(1-cos)*KK
    return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    # theta=np.arccos(1/2*(np.trace(R)-1))
    # rx=(R-np.transpose(R))/(2*sin)
    w,W=np.linalg.eig(np.transpose(R))
    i=np.where(abs(np.real(w)-1)<10e-14)[0]
    if len(i)==0:
        raise ValueError('no unit eignvector corresponding to eigenvalue 1')
    else:
        K=np.real(W[:,i[-1]]).squeeze()
    cos=1/2*(np.trace(R)-1)

    angle=np.arccos(1/2*(np.trace(R)-1))

    r=K*angle
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    residuals = None
    N=p1.shape[0]

    M=x.shape[0]
    P=x[:M-6].reshape(-1,3)
    r=x[M-6:M-3]
    t=np.vstack(x[M-3:M]) # t:3*1
    R=rodrigues(r)
    M2=np.hstack([R,t])

    C1=K1.dot(M1)
    C2=K2.dot(M2)

    p1_hat=np.zeros(p1.shape)
    p2_hat=np.zeros(p1.shape)
    err=np.zeros(N)
    for i in range(N):
        point=np.append(P[i,:],1)
        p1_hat_i=np.dot(C1,point)
        p1_hat_i=p1_hat_i/p1_hat_i[-1]
        p1_hat_i=p1_hat_i[0:-1]
        p2_hat_i=np.dot(C2,point)
        p2_hat_i=p2_hat_i/p2_hat_i[-1]
        p2_hat_i=p2_hat_i[0:-1]
        p1_hat[i,:]=p1_hat_i
        p2_hat[i,:]=p2_hat_i
    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])]).reshape(-1,1)   #residuals: 4N*1
    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    M2, P = None, None
    R=M2_init[:,0:3]
    r=invRodrigues(R)
    t=M2_init[:,3]
    x=np.hstack((P_init.flatten(),r.flatten(),t))   #x: 3N+6:   P_init.flatten: 3*N  r.flatten: 3*1, t: 3*1
    pack=(K1, M1, p1, K2, p2)
    err_original=error(x, K1, M1, p1, K2, p2)
    print('original error',err_original)
    print('start optimize')
    response = scipy.optimize.minimize(fun=error, args=pack, x0=x)
    print('finish optimize')
    bestx=response['x']

    M=bestx.shape[0]
    P_optimized=bestx[:M-6]
    r=bestx[M-6:M-3]
    t=bestx[M-3:M]
    R=rodrigues(r)
    M2=np.hstack((R,t[:,None]))
    err_final=error(bestx, K1, M1, p1, K2, p2)
    print('optimized error',err_final)
    return M2,P_optimized

def error(x, K1, M1, p1, K2, p2):
        value=rodriguesResidual(K1, M1, p1, K2, p2, x)
        error=sum(value**2)
        # print('error',error)
        return error
