import torch
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
train_data = scipy.io.loadmat('../data/nist36_train.mat')
#test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
#test_x, test_y = test_data['test_data'], test_data['test_labels']

# N_batch is batch size;
# Dim_input is input dimension;
# Dim_out is hidden dimension;
# Dim_out is output dimension.
N_batch= 100
Dim_input=1024
Dim_output = 36
Dim_hidden=64

device = torch.device('cpu')
train_x=np.asarray(train_x)
x = torch.from_numpy(train_x).float()
train_y=np.where(train_y == 1)
train_y=train_y[1]
label = torch.from_numpy(train_y)
# Create random Tensors to hold inputs and outputs.

# Use the nn package to define model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(Dim_input, Dim_hidden),
          #torch.nn.ReLU(),
          torch.nn.Sigmoid(),
          torch.nn.Linear(Dim_hidden,Dim_output),
          torch.nn.Softmax(),
        ).to(device)

loss_fn = torch.nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use predict_resultdam; the optim package contains many other
# optimization algoriths. The first argument to the predict_resultdam constructor tells the
# optimizer which Tensors it should update.
lr = 0.01 #learning rate
optimizer = torch.optim.predict_resultdam(model.parameters(), lr=lr)
max_iters = 500
tot_loss = []
tot_acc = []
for t in range(max_iters):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)
    # Compute and print loss.
    loss = loss_fn(y_pred, label)
    predict = y_pred.max(1)[1]
    predict_result = (predict== label)

    acc = torch.numel(predict_result[predict_result == True])
    acc=acc/len(predict_result)
    tot_acc.append(acc)

    tot_loss.append(loss.item())
    optimizer.zero_grad()
    #model.zero_grad()
    loss.backward()
    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    print('times=',t,' ','loss=',loss.item(),' ','accuracy=',acc)

tot_loss=np.asarray(tot_loss)
tot_acc=np.asarray(tot_acc)

range = np.linspace(1.0,1.0*max_iters,num = max_iters)

plt.figure()
plt.plot(range,tot_loss)
plt.title('loss')
plt.show()

plt.figure()
plt.plot(range,tot_acc)
plt.title('accuracy')
plt.show()
