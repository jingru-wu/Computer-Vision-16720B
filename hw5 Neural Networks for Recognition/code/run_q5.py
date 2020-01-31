import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
# defualt parameters settings
batch_size = 36
# learning_rate = 3e-5 # defualt settings
learning_rate = 2e-6
hidden_size = 32
lr_rate = 20
batches,_ = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,1024,params,'output')
params['M_layer1'],params['M_output'] = 0,0

loss=[]
# with default settings, you should get loss < 150 and accuracy > 80%
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        batch_h1 = forward(xb,params,'layer1',relu)
        batch_h2 = forward(batch_h1,params,'hidden',relu)
        batch_h3 = forward(batch_h2,params,'hidden2',relu)
        batch_probs = forward(batch_h3,params,'output',sigmoid)
        # loss
        # be sure to add loss and accuracy to epoch totals
        #batch_loss, batch_acc = compute_loss_and_acc(xb, batch_probs)
        #total_loss = total_loss + batch_loss
        diff=(batch_probs - xb)
        batch_loss = diff**2
        total_loss = total_loss+batch_loss.sum()
        batch_delta1 = 2*diff

        # backward
        batch_delta2 = backwards(batch_delta1,params,'output')  #34*32
        batch_delta3 = backwards(batch_delta2,params,'hidden2',relu_deriv) # 34*32
        batch_delta4 = backwards(batch_delta3,params,'hidden',relu_deriv) # 34*32
        backwards(batch_delta4,params,'layer1',relu_deriv)

        # apply gradient

        '''params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Whidden'] = params['Whidden'] - learning_rate*params['grad_Whidden']
        params['Whidden2'] = params['Whidden2'] - learning_rate*params['grad_Whidden2']
        params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
        params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
        params['bhidden'] = params['bhidden'] - learning_rate*params['grad_bhidden']
        params['bhidden2'] = params['bhidden2'] - learning_rate*params['grad_bhidden2']
        params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']'''

        #update parameters
        params['M_layer1'] = 0.9*params['M_layer1'] - learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] = params['Wlayer1'] + params['M_layer1']
        params['M_hidden'] = 0.9*params['M_hidden'] - learning_rate*params['grad_Whidden']
        params['Whidden'] = params['Whidden'] + params['M_hidden']
        params['M_hidden2'] = 0.9*params['M_hidden2'] - learning_rate*params['grad_Whidden2']
        params['Whidden2'] = params['Whidden2'] + params['M_hidden2']
        params['M_output'] = 0.9*params['M_output'] - learning_rate*params['grad_Woutput']
        params['Woutput'] = params['Woutput'] + params['M_output']

        params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
        params['bhidden'] = params['bhidden'] - learning_rate*params['grad_bhidden']
        params['bhidden2'] = params['bhidden2'] - learning_rate*params['grad_bhidden2']
        params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
    loss.append(total_loss)
loss=np.asarray(loss)

plt.figure()
epoch = np.linspace(1.0,100.0,num = 100)
plt.plot(epoch,loss)
plt.title('train loss')
plt.show()

# visualize some results

# Q5.3.1
import matplotlib.pyplot as plt
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(np.transpose(xb[i].reshape(32,32)))
    plt.subplot(2,5,i+6)
    plt.imshow(np.transpose(out[i].reshape(32,32)))
plt.show()


valid_y = valid_data['valid_labels']

# visulize vildation
idx = [0,1,100,101,200,201,300,301,400,401] # manually choose first 5 class and first2 validation images for each class
xb = valid_x[idx]
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(10):
    plt.figure()
    plt.subplot(2,1,1),plt.imshow(np.transpose(xb[i].reshape(32,32)))
    plt.title('validation original letter')
    plt.subplot(2,1,2),plt.imshow(np.transpose(out[i].reshape(32,32)))
    plt.title('validation encode letter')
    plt.show()

# evaluate PSNR
# Q5.3.2
from skimage.measure import compare_psnr as psnr

h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out_label = forward(h3,params,'output',sigmoid)

n_valid=(valid_x.shape[0])
psnr_list = []
for i in range(n_valid):
    psnr_single = psnr(valid_x[i,:],out_label[i,:])
    psnr_list.append(psnr_single)
psnr_average = np.array(np.mean(psnr_list))
print('average PSNR across all validation images:',psnr_average)

