import numpy as np
import matplotlib.pyplot as plt
import cv2
import BRIEF
import keypointDetect
import scipy.ndimage
import scipy.signal

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = BRIEF.makeTestPattern()
    # test briefLite
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # locs, desc = briefLite(im)
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.plot(locs[:,0], locs[:,1], 'r.')
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)
    # test matches
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    # im1 = cv2.imread('../data/incline_R.png')
    # im2 = cv2.imread('../data/incline_L.png')
    locs1, desc1 = BRIEF.briefLite(im1)
    (cX, cY)= (im1.shape[1]/ 2,im1.shape[0]/ 2)
    num_correct=[]
    x= [0,10,20,30,40,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360]
    for a in x:
        M=cv2.getRotationMatrix2D((cX,cY),angle=a,scale=1.0)
        im2=cv2.warpAffine(im1, M,dsize=(im1.shape[1],im1.shape[0]))
        locs2, desc2 = BRIEF.briefLite(im2)
        matches = BRIEF.briefMatch(desc1, desc2)
        num_correct.append(len(matches))
    num_correct=np.asarray(num_correct)
    print(num_correct)
    plt.bar(x,num_correct,width=10)
    plt.title('Bar chat of correct match number VS degree from 0 to 360')
    plt.xlabel('degree')
    plt.ylabel('number of correct match pairs')
    plt.show()
#    BRIEF.plotMatches(im1,im2,matches,locs1,locs2)


