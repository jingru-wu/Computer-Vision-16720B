# write your script here, we recommend the above libraries for making your animation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import LucasKanadeBasis
import LucasKanade
import cv2
# write your script here, we recommend the above libraries for making your animation
#  Displaying at frames 1,100,200,300,400
if __name__ == '__main__':
    # sylv=np.load('../data/sylvseq.npy')
    # x1=101
    # x2=155
    # y1=61
    # y2=107
    # number_image = sylv.shape[2]
    # rect=np.asarray([x1,y1,x2,y2])
    # rects1=np.zeros([number_image,4])
    # rects1[0]=rect
    # bases=np.load('../data/sylvbases.npy')
    # for i in range(number_image-1):
    #     It = sylv[:,:,i]
    #     It1 = sylv[:,:,i+1]
    #     p = LucasKanadeBasis.LucasKanadeBasis(It, It1, rect,bases)
    #     x1 = x1+p[0]
    #     x2 = x2+p[0]
    #     y1 = y1+p[1]
    #     y2 = y2+p[1]
    #     print('image number:',i)
    #     rect=np.array([x1,y1,x2,y2])
    #     rects1[i+1,:]=rect
    # np.save('../data/sylvseqrects.npy', rects1)
    #
    # x1=101
    # x2=155
    # y1=61
    # y2=107
    # rect=np.asarray([x1,y1,x2,y2])
    # rects2=np.zeros([number_image,4])
    # for i in range(number_image-1):
    #     It = sylv[:,:,i]
    #     It1 = sylv[:,:,i+1]
    #     p = LucasKanade.LucasKanade(It, It1, rect)
    #     x1 = x1+p[0]
    #     x2 = x2+p[0]
    #     y1 = y1+p[1]
    #     y2 = y2+p[1]
    #     print('image number',i)
    #     rect=np.array([x1,y1,x2,y2])
    #     rects2[i+1,:]=rect
    #     np.save('../data/sylvseqrects_kl.npy', rects2)

## test Rectangles
    sylv=np.load('../data/sylvseq.npy')
    rects1=np.load('../data/sylvseqrects.npy')       # rect Bases rectangle
    # rect2=np.load('../data/sylvseqrects_kl.npy')    # rect Lucas rectangle
    rects2=np.load('../data/sylvrect_tc.npy')        # rect with template correction
    number_image = sylv.shape[2]

    for i in range(number_image-1):
# rect=np.asarray([x1,y1,x2,y2])

        if i==1 or i==200 or i==300 or i==350 or i==400:
            fig,ax = plt.subplots(1)
            ax.imshow(sylv[:,:,i],cmap=plt.get_cmap('gray'))
            mark1=patches.Rectangle([rects1[i,0],rects1[i,1]],rects1[i,2]-rects1[i,0],rects1[i,3]-rects1[i,1],linewidth=1,edgecolor='y',facecolor='none')
            mark2=patches.Rectangle([rects2[i,0],rects2[i,1]],rects2[i,2]-rects2[i,0],rects2[i,3]-rects2[i,1],linewidth=1,edgecolor='g',facecolor='none')
            # mark3=patches.Rectangle([rect3[i,0],rect3[i,1]],rect3[i,2]-rect3[i,0],rect3[i,3]-rect3[i,1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(mark1)
            ax.add_patch(mark2)
            # ax.add_patch(mark3)
            # plt.pause(0.05)
            plt.title('frame: %i'%i)

            plt.show()

    print('finished')
