import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 30
learning_rate = 0.01
hidden_size = 64

batches,bpos = get_random_batches(train_x,train_y,batch_size)
# batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')
accuracy = []

loss = []
valid_acc = []
valid_loss = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    
    acc = []
    
    predict_label = np.zeros([train_y.shape[0],train_y.shape[1]])
    n = 0
    for xb,yb in batches:
        batch_1 = forward(xb,params,'layer1')
        batch_single = forward(batch_1,params,'output',softmax)
        # loss
        #probs.append(batch_single)
        # be sure to add loss and accuracy to epoch totals
        batch_loss, batch_acc = compute_loss_and_acc(yb, batch_single)
        total_loss = total_loss + batch_loss
        
        acc.append(batch_acc)
        batch_d1 = batch_single
        #confusion matrix
        predict_label[bpos[n*batch_size:(n+1)*batch_size].astype(int)] = batch_single
        n = n+1
        yb_idx=np.where(yb == 1)[1]
        batch_d1[np.arange(len(batch_single)),yb_idx] -= 1
        # backward
        batch_d2 = backwards(batch_d1,params,'output',linear_deriv)
        backwards(batch_d2,params,'layer1',sigmoid_deriv)
        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
        params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
        params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']
    total_acc = np.mean(acc)
    accuracy.append(total_acc)
    loss.append(total_loss)
    #calculate validation
    
    valid_hidden1 = forward(valid_x,params,'layer1')
    val_probs = forward(valid_hidden1,params,'output',softmax)
    # be sure to add loss and accuracy to epoch totals
    val_loss, val_acc = compute_loss_and_acc(valid_y,val_probs)
    valid_loss.append(val_loss)
    valid_acc.append(val_acc)
    # print('Validation accuracy: ',val_acc)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
# run on validation set and report accuracy! should be above 75%
epoch = np.linspace(1.0,1.0*max_iters,num = max_iters)
plt.figure(1)
plt.plot(epoch,accuracy,label = 'train accuracy')
plt.plot(epoch,valid_acc,label = 'valid accuracy')
plt.title('accuracy')
plt.legend()

plt.figure(2)
plt.plot(epoch,loss,label = 'train loss')
plt.plot(epoch,valid_loss,label = 'valid loss')
plt.title('loss')
plt.legend()
plt.show()
valid_acc = valid_acc[-1]

# print('Validation accuracy: ',valid_acc)
#test accuracy
test_h1 = forward(test_x,params,'layer1')
test_probs = forward(test_h1,params,'output',softmax)
test_loss, test_acc = compute_loss_and_acc(test_y,test_probs)
print('test accuracy:',test_acc)



if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
weight_2 = params['W' + 'layer1'] # first weight
[im_size,h_size] = np.shape(weight_2)

plt.figure(3)
fig_1 = plt.figure(3, (8., 8.))
plt.title('initial network')
grid_1 = ImageGrid(fig_1,111,nrows_ncols=(8,8),axes_pad=0.01)

for i in range(h_size):
    l = np.sqrt(6.0 / (1024 + h_size))
    weight_1 = np.random.uniform(-l,l,[1024,h_size])
    im1 = weight_1[:,i]
    im1=im1.reshape(32,32)
    grid_1[i].imshow(im1)  # The AxesGrid object work as a list of axes.
plt.show()

plt.figure(4)
fig_2 = plt.figure(4, (8., 8.))
plt.title('learnt network')
grid_2 = ImageGrid(fig_2,111,nrows_ncols=(8,8),axes_pad=0.01)

for i in range(h_size):
    im2 = weight_2[:,i]
    im2=im2.reshape(32,32)
    grid_2[i].imshow(im2)  # The AxesGrid object work as a list of axes.
plt.show()

# Q3.1.4 Visualize the confusion matrix
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for i in range(train_y.shape[0]):
    index_x = np.where(train_y[i,:]==1)[0]
    index_y = np.where(predict_label[i,:] == max(predict_label[i,:]))[0]
    index_x=index_x.round()
    index_y=index_y.round()
    confusion_matrix[index_x,index_y] = confusion_matrix[index_x,index_y] + 1

import string
plt.imshow(confusion_matrix)
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
print('finished Run Q3')
