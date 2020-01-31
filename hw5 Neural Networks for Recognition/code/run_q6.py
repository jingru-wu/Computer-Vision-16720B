import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

# we don't need labels now!
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

dim = 32
# do PCA
[U,s,V] = np.linalg.svd(train_x)
V=np.transpose(V)

# rebuild a low-rank version
lrank = dim

#projection_matrix
pro_matrix = np.dot(V[:,:lrank],np.transpose(V[:,:lrank]))

# rebuild it
recon  = np.dot(test_x, pro_matrix)
index=[0,1,50,51,100,101,150,151,200,201]
test_x=test_x[index]
recon=recon[index]
for i in range(5):
    plt.subplot(2,2,1)
    plt.imshow(test_x[2*i].reshape(32,32).T)
    plt.subplot(2,2,3)
    plt.imshow(recon[2*i].reshape(32,32).T)
    plt.subplot(2,2,2)
    plt.imshow(test_x[2*i+1].reshape(32,32).T)
    plt.subplot(2,2,4)
    plt.imshow(recon[2*i+1].reshape(32,32).T)
    plt.show()

# build valid dataset
recon_valid = valid_x.dot(pro_matrix)

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print('average PSNR across all validation images:',np.array(total).mean())
