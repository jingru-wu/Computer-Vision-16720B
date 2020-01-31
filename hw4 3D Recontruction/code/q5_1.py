import numpy as np
import submission as sub
import helper
import matplotlib.pyplot as plt
def ransacF(pts1, pts2, M):
    num_iter=5000
    tol=10e-3
    #create a set of random index to match points
    randIdx=np.random.randint(0,pts1.shape[0]-1,size=num_iter*8,dtype=int)
    randIdx=randIdx.reshape(num_iter,8)
    max_inliers=0
    for i in range(num_iter):
        index=randIdx[i,:]
        P1=pts1[index,:]
        P2=pts2[index,:]
        F=sub.eightpoint(P1,P2,M)

        num_inliers=0
        inliers_points=np.repeat(False,len(pts1))
        for m in range(len(pts1)):
            p2=np.vstack([pts2[m,0],pts2[m,1],1])
            epipolar_line = np.dot(F,p2)
            # a = epipolar_line[0]
            # b = epipolar_line[1]
            # c = epipolar_line[2]
            p1=np.hstack([pts1[m,0],pts1[m,1],1])
            diff= abs(np.dot(p1,epipolar_line))
            if diff<tol:
                num_inliers=num_inliers+1
                inliers_points[m]=True
        if num_inliers>max_inliers:
            max_inliers=num_inliers
            inliers=inliers_points
            F_best=F
        print('times:',i)
    print('max num of inlier points',max_inliers)



    num_iter=5000
    tol=3*10e-4
    #create a set of random index to match points
    randIdx=np.random.randint(0,pts1.shape[0]-1,size=num_iter*7,dtype=int)
    randIdx=randIdx.reshape(num_iter,7)
    max_inliers=0
    for i in range(num_iter):
        print('times:',i)
        index=randIdx[i,:]
        P1=pts1[index,:]
        P2=pts2[index,:]
        Farray=sub.sevenpoint(P1,P2,M)
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
    print('max num of inlier points',max_inliers)
    return F_best,inliers








if __name__ == '__main__':
    data=np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    pts1=data['pts1']
    pts2=data['pts2']

    M=650
    [F, inliers]=sub.ransacF(pts1,pts2,M)
    print('best F:\n',F)
    # print('inliers=',inliers)
    pts1_inliers=pts1[inliers,:]
    pts2_inliers=pts2[inliers,:]
    F = sub.eightpoint(pts1_inliers, pts2_inliers, M)
    helper.displayEpipolarF(im1, im2, F)


# # directly run eightpoint on noisy correspondance
#     F = sub.eightpoint(pts1, pts2, M)
#     helper.displayEpipolarF(im1, im2, F)
