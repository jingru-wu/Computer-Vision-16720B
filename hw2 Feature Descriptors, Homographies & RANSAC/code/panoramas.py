import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import scipy
def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    warp_im =skimage.transform.warp(im2,H2to1,output_shape=(im2.shape[0],1000))
    # warp_p=cv2.cvtColor(warp_im,cv2.COLOR_RGB2BGR)
    pano_im=warp_im
    pano_im[:im1.shape[0],:im1.shape[1]]=im1
    #
    plt.imshow(pano_im*255),plt.title('warped_image')
    plt.show()

    # cv2.imwrite('../results/q6_1.jpg',warp_im)
    #
    # cv2.imshow('warped image', warp_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    im1=im1/255
    im2=im2/255

    im2_Y = im2.shape[0]    #Height range
    im2_X = im2.shape[1]
    corner = np.matrix([[0,im2_X, 0, im2_X],[0, 0, im2_Y, im2_Y],[1,   1,  1,  1]])
    warped_corner = np.dot(H2to1,corner)               # warp im2 corners into im1 reference frame
    warped_corner = warped_corner/warped_corner[2,:]
    maxY = max([im1.shape[0],max(np.transpose(warped_corner[1,:]))])
    minY= min([0,min(np.transpose(warped_corner[1,:]))])
    maxX = max([im1.shape[1],max(np.transpose(warped_corner[0,:]))])
    minX = min([0,min(np.transpose(warped_corner[0,:]))])
    NewCorner_points=np.array([[minX,maxX,minX,maxX],[minY,minY,maxY,maxY]])
    ratio = (maxY - minY)/(maxX - minX)         #ratio: fraction between height and width
    width=1000
    height = int(width*ratio)
#calculate height with input width
    out_size = (width,height)
    print('out_size:',out_size)

    s = width/(maxX - minX)         #s: scale = outsize / inputsize( max size in im1 and warpped im2)_
    scale_M = [[s,0,0],[0,s,0],[0,0,1]]
    trans_M = [[1,0,-minX],[0,1,-minY],[0,0,1]]
    M=np.matrix(np.dot(scale_M,trans_M))

    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, np.dot(M,H2to1),out_size)


    mask1 = np.zeros((im1.shape[0],im1.shape[1],3))
    mask1[0,:] = 1
    mask1[-1,:] = 1
    mask1[:,0] = 1
    mask1[:,-1] = 1
    mask1 = scipy.ndimage.morphology.distance_transform_edt(1-mask1)
    mask1 = mask1/mask1.max()
    mask1[np.isnan(mask1)]=1

    mask2 = np.zeros((im2.shape[0],im2.shape[1],3))
    mask2[0,:] = 1
    mask2[-1,:] = 1
    mask2[:,0] = 1
    mask2[:,-1] = 1
    mask2 = scipy.ndimage.morphology.distance_transform_edt(1-mask2)
    mask2 = mask2/mask2.max()
    mask2[np.isnan(mask2)]=1


    mask1_warp= cv2.warpPerspective(mask1, M, out_size)
    mask2_warp = cv2.warpPerspective(mask2, np.dot(M,H2to1),out_size)

    maskweight1=mask1_warp/(mask1_warp+mask2_warp)
    maskweight1[np.isnan(maskweight1)]=1
    maskweight2=mask2_warp/(mask1_warp+mask2_warp)
    maskweight2[np.isnan(maskweight2)]=1

    pano_im=np.multiply(warp_im1,maskweight1)+np.multiply(warp_im2,maskweight2)

    #
    # im1_p=cv2.cvtColor(im1,cv2.COLOR_RGB2BGR)
    # im2_p=cv2.cvtColor(im2,cv2.COLOR_RGB2BGR)
    # warp_p=cv2.cvtColor(pano_im,cv2.COLOR_RGB2BGR)
    # plt.subplot(1,3,1),plt.imshow(im1_p)
    # plt.subplot(1,3,2),plt.imshow(im2_p)
    # plt.subplot(1,3,3),plt.imshow(warp_p)
    # plt.show()
    return pano_im

def generatePanorama(im1, im2):
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    #

    # H2to1=[[6.50084832e-01,-3.13720032e-02,3.63107019e+02],
    #     [-8.23718229e-02,8.81596960e-01,-1.78663919e+01],
    #     [-3.61757383e-04,-5.74087127e-06,1.00000000e+00]]
    # H2to1=np.matrix(H2to1)

    H2to1=np.load('../results/6_1.npy')

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print('bestH:\n',H2to1)

    return pano_im


if __name__ == '__main__':
    im1 = skimage.io.imread('../data/p2.png')
    im2 = skimage.io.imread('../data/p1.png')
    # print('image1 shape:',im1.shape)
    # print('image2 shape:',im2.shape)
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    #

# ## Q6.1
#     H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    H2to1=np.load('../results/6_1.npy')
    print('bestH:\n',H2to1)
    np.save('../results/6_1.npy',H2to1)
# test imageStitching()
    imageStitching(im1, im2, H2to1)
    print('Q6.1 finished')



# ## Q6.2

    # H2to1=[[6.50084832e-01,-3.13720032e-02,3.63107019e+02],
    #     [-8.23718229e-02,8.81596960e-01,-1.78663919e+01],
    #     [-3.61757383e-04,-5.74087127e-06,1.00000000e+00]]
#     # H2to1=np.matrix(H2to1)
#
#     H2to1=np.load('../results/q6_1.npy')
#     pano_im = imageStitching_noClip(im1, im2, H2to1)
#     cv2.imwrite('../results/q6_2_pan.jpg',pano_im*255)
#     cv2.imshow('panoramas', pano_im)
#     cv2.imwrite('../results/panoImg.png', pano_im*255)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print('Q6.2 finished')
# #
# # Q6.3
#     pano_im = generatePanorama(im1, im2)
#     cv2.imwrite('../results/q6_3.jpg',pano_im*255)
#     cv2.imshow('panoramas', pano_im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print('Q6.3 finished')
