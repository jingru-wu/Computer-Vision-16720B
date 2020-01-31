import numpy as np
import cv2

import matplotlib.pyplot as plt
import scipy.ndimage
def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
#Q1.2
def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    for i in range(len(levels)-1):
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    Dx=cv2.Sobel(DoG_pyramid,-1,1,0,ksize=3, borderType=cv2.BORDER_CONSTANT)
    Dy=cv2.Sobel(DoG_pyramid,-1,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT)
    Dxx=cv2.Sobel(Dx,-1,1,0,ksize=3,borderType=cv2.BORDER_CONSTANT)
    Dxy=cv2.Sobel(Dx,-1,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT)
    Dyy=cv2.Sobel(Dy,-1,0,1,ksize=3,borderType=cv2.BORDER_CONSTANT)
    Dyx=cv2.Sobel(Dy,-1,1,0,ksize=3,borderType=cv2.BORDER_CONSTANT)

    Tra=abs(Dxx+Dyy)
    Det=abs(Dxx*Dyy-Dxy*Dyx)
    principal_curvature=Tra**2/Det

    # plt.subplot(2,2,1),plt.imshow(DoG_pyramid[:,:,1],cmap = 'gray')
    # plt.subplot(2,2,2),plt.imshow(Dx[:,:,1],cmap = 'gray')
    # plt.subplot(2,2,3),plt.imshow(Dxy2[:,:,1],cmap = 'gray')
    # plt.subplot(2,2,4),plt.imshow(Dxy[:,:,1],cmap = 'gray')
    # plt.show()
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    DoG=abs(DoG_pyramid)
    max_array=np.zeros(DoG.shape)
    min_array=np.zeros(DoG.shape)
    for i in DoG_levels:
        max_array[:,:,i]=scipy.ndimage.maximum_filter(DoG[:,:,i],size=3,mode='reflect')
        min_array[:,:,i]=scipy.ndimage.minimum_filter(DoG[:,:,i],size=3,mode='reflect')
    max_position=(DoG==max_array)
    min_position=(DoG==min_array)
    locsDoG=[]
    for i in DoG_levels:
        for x in range(len(DoG_pyramid[:,0,0])):
            for y in range(len(DoG_pyramid[0,:,0])):
                if 0<i<4:
                    if max_position[x,y,i]:
                        if DoG[x,y,i]<DoG[x,y,i+1] or DoG[x,y,i]<DoG[x,y,i-1]:
                            max_position[x,y,i]=False
                    if min_position[x,y,i]:
                        if DoG[x,y,i]>DoG[x,y,i+1]or DoG[x,y,i]>DoG[x,y,i-1]:
                            min_position[x,y,i]=False
                if i==0:
                    if max_position[x,y,i]:
                        if DoG[x,y,i]<DoG[x,y,i+1]:
                            max_position[x,y,i]=False
                    if min_position[x,y,i]:
                        if DoG[x,y,i]>DoG[x,y,i+1]:
                            min_position[x,y,i]=False
                if i==4:
                    if max_position[x,y,i]:
                        if DoG[x,y,i]<DoG[x,y,i-1]:
                            max_position[x,y,i]=False
                    if min_position[x,y,i]:
                        if DoG[x,y,i]>DoG[x,y,i-1]:
                            min_position[x,y,i]=False
                if DoG[x,y,i]<th_contrast or principal_curvature[x,y,i]>th_r:
                #if DoG[x,y,i]<th_contrast:   #without edge supression
                     max_position[x,y,i]=False
                     min_position[x,y,i]=False
                if max_position[x,y,i] or min_position[x,y,i]:
                    a=np.asarray([y,x,i])
                    locsDoG.append(a)
    locsDoG=np.asarray(locsDoG)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    gauss_pyramid = createGaussianPyramid(im,sigma0,k,levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    #plot keypoints


    return locsDoG, gauss_pyramid



if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im = cv2.imread('../data/incline_R.png')
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255

# #Q1.1 test
#     im_pyr = createGaussianPyramid(im)
#     displayPyramid(im_pyr)
# # #Q1.2 test DoG pyramid
#     DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
#     displayPyramid(DoG_pyr)

# #Q1.3 test compute principal curvature
#     pc_curvature = computePrincipalCurvature(DoG_pyr)

# #Q1.4 test get local extrema
#     th_contrast = 0.03
#     th_r = 12
#     locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
#     plt.imshow(im,cmap=plt.get_cmap('gray'))
#     plt.plot(locsDoG[:,0],locsDoG[:,1],'b.')
#     plt.show()
#
#  #Q1.5  test DoG detector
#     locsDoG, gaussian_pyramid = DoGdetector(im)
#     plt.imshow(im,cmap=plt.get_cmap('gray'))
#     plt.plot(locsDoG[:,0],locsDoG[:,1],'b.')
#     plt.title('DoG detector')
#     plt.show()
#
