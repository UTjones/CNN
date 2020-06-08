from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#A new neural network set up for cool people
class cpNet(nn.Module):
   
    def __init__(self):
        super(cpNet, self).__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, 
                               kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                               kernel_size = 3)

        
        self.pool = nn.MaxPool2d(2,2)
    
        self.fc1 = nn.Linear(14*14*64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = nn.ReplicationPad2d(1)(x)
        x = self.pool(F.relu(self.conv1(x)))
        
        x = nn.ReplicationPad2d(1)(x)
        
        x = F.relu(self.conv2(x))
        

        x = x.view(-1, 14*14*64)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


network = cpNet()
import torch.optim as optim


optimizer = optim.Adam(network.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

train_X = torch.Tensor(train.iloc[:, 1:].values).view(-1, 28, 28)
trainList = []
for i in train.iloc[:, 0].values:
    trainList.append(np.eye(10)[i])

train_Y = torch.tensor(trainList).float()

BATCH_SIZE = 32
EPOCHS = 8
loss_values = []
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train), BATCH_SIZE)):
        batch_x = train_X[i: i+BATCH_SIZE].view(-1, 1, 28, 28)
        batch_y = train_Y[i: i+BATCH_SIZE]
        
        network.zero_grad()
        
        outputs = network(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    loss_values.append((epoch, loss))  

test_X = torch.Tensor(train.iloc[:, 1:].values).view(-1, 28, 28)
testList = []
for i in train.iloc[:, 0].values:
    testList.append(np.eye(10)[i])

test_Y = torch.tensor(testList).float()

correct = 0
total = 0
count = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_Y[i])
        net_out = network(test_X[i].view(-1, 1, 28, 28))[0]
        predicted_class = torch.argmax(net_out)
    
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 4))

x, y = [], []
for i in loss_values:
    x.append(i[0])
    y.append(i[1].item())
    
plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
