import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    aList=[]
    for i in range(p1.shape[1]):
        x=p1[0,i]
        y=p1[1,i]
        u=p2[0,i]
        v=p2[1,i]
        a1=[0, 0, 0,-u,-v,-1,y*u,y*v,y]
        a2=[u,v,1,0,0,0,-x*u,-x*v,-x]
        aList.append(a1)
        aList.append(a2)
    A=np.matrix(aList)
    (U,S,V) = np.linalg.svd(A)
    V=np.transpose(V)
    H2to1 = np.reshape(V[:,8],(3,3))
    H2to1=H2to1/H2to1[2,2]

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    pointset1=np.transpose(locs1[matches[:,0]][:,:2])
    pointset2=np.transpose(locs2[matches[:,1]][:,:2])
    #create a set of random index to match points
    randIdx=np.random.randint(0,pointset1.shape[1]-1,size=num_iter*4,dtype=int)
    randIdx=randIdx.reshape(num_iter,4)
    TransPoint=np.zeros([2,matches.shape[0]])
    num=[]
    H_list=[]
    for i in range(num_iter):
        index=randIdx[i,:]
        p1=pointset1[:,index]
        p2=pointset2[:,index]
        H=computeH(p1, p2)
        H_list.append(H)
        num_lniers=0
        for m in range(matches.shape[0]):
            p=np.transpose(np.matrix([pointset2[0,m],pointset2[1,m],1]))
            trans_point=np.dot(H,p)
            trans_point=trans_point/trans_point[2,0]
            TransPoint=np.transpose([trans_point[0,0],trans_point[1,0]])
            dist = np.linalg.norm(pointset1[:,m]-TransPoint)
            if dist<tol:

                num_lniers=num_lniers+1
        num.append(num_lniers)
    num=np.asarray(num)
    maxnum_index=np.argmax(num)
    bestH=H_list[maxnum_index]
    print('max num of inlier points',max(num))
    print('best H:\n',bestH)
    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    print('length of match:',len(matches))
#Q4.1 test computeH
    # point1=np.transpose(locs1[matches[:,0]][:,:2])
    # point2=np.transpose(locs2[matches[:,1]][:,:2])
    # H=computeH(point1,point2)
    # print('H:\n',H)
#Q5.1
    bestH=ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
