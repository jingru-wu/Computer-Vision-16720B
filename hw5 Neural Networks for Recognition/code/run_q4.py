import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Q4.2
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    boxes_list, bw = findLetters(im1)
    boxes_list = np.asarray(boxes_list)
    plt.imshow(bw,cmap = plt.cm.gray)
    for box in boxes_list:
            y1, x1, y2, x2 = box
            
            rectangle = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rectangle)

    plt.tight_layout()
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    row_list = []
    for i in range(len(boxes_list)):
        y1, x1, y2, x2 = boxes_list[i]
        row_list.append(y1)
    num_row = 0
    i = row_list[0]
    lines = []
    m = 0
    for j in range(len(row_list)):
        if row_list[j] - i > 100:# test difference between two value
            each_line = boxes_list[m:j]
            num_row += 1
            lines.append(each_line)
            m = j
        if j == len(row_list)-1:# for last line
            lines.append(boxes_list[m:j])
        i = row_list[j]
    num_row = num_row+1#for the last row
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])#label
    params = pickle.load(open('q3_weights.pickle','rb'))
    #total_text = []
    #bw = bw[:, ::-1]
    text=''
    for i in range(num_row):
        #total_text = []
        one_line = lines[i] #this line
        one_line = one_line[one_line[:,1].argsort()]
        num_element = len(one_line)# how many element in each row
        diss = np.diff(one_line[:,1])
        threshold = np.mean(diss)
        for j in range(num_element):
            pos = one_line[j,:]
            data = bw[int(pos[0]):int(pos[2]),int(pos[1]):int(pos[3])]
            column_shape = pos[3] - pos[1]
            row_shape = pos[2] - pos[0]#get size of row and column
            #padding
            if column_shape > row_shape:
                    c=column_shape
                    r=row_shape
            else:
                    c=row_shape
                    r=column_shape

            pad_shape = int(c/5)
            patch_1 = (c + 2*pad_shape - r) // 2
            patch_2 = c + 2*pad_shape - patch_1 - r

            if column_shape > row_shape:
                    data = np.pad(data,[(patch_1,patch_2),(pad_shape,pad_shape)],mode = 'constant',constant_values=1)
            else:   data = np.pad(data,[(pad_shape,pad_shape),(patch_1,patch_2)],mode = 'constant',constant_values=1)

            data = skimage.transform.resize(data,(32,32))
            data = data < data.max()
            data = np.transpose(data==0).reshape(1,1024)
            #data = data.flatten()
            h1 = forward(data,params,'layer1')
            probs = forward(h1,params,'output',softmax)
            predict_label = np.argmax(probs,axis = 1)
            predict_label = np.int(predict_label)
            if j != (num_element-1):
                pos_next = one_line[j+1,:]

                if np.abs(pos[1] - pos_next[1]) < 1.5*threshold:#if no need to add space
                    text = text + letters[predict_label]
                else:

                    text = text + letters[predict_label]
                    text = text + ' '#for next

            if j == (num_element-1):
                    text = text + letters[predict_label]
        text = text + '\n'
    print('image',img)
    print(text)
print('finished Run Q4')
