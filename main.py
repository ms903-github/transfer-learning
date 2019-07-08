import glob
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocessor, DatalisttoDataset
from fetch_data import get_data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict

batch_size = 100
num_epoch = 50
data, label = get_data(10, num_ratio = 10, domain = "target", mode = "choiced")
print("preprocess finished")
dataset = DatalisttoDataset(data, label, transform = None)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) 
        self.conv2 = nn.Conv2d(32, 32, 3) 
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3) 
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)           # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(36992, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))                # 152x152x3 -> 150x150x32
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))     #150x150x32 -> 148x148x32 -> 74x74x32
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))                #74x74x32 -> 72x72x64
        x = self.pool(F.relu(self.conv4(x)))           #72x72x64 -> 70x70x32 -> 35x35x32
        x = x.view(-1, 36992)               #35x35x32 -> 39200
        x = F.relu(self.fc1(x))                  #39200 -> 128
        x = self.fc2(x)                          #128 -> 10
        return x

net = Net().to(device)
net.load_state_dict(torch.load("net.model"))
#checkpoint = torch.load("net.model")
#state_dict = checkpoint
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:]
#    new_state_dict[name] = v
#net.load_state_dict(new_state_dict)

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(num_epoch):
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(data)
        loss = loss_func(pred, label)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("epoch {} has finished".format(epoch))
        with torch.no_grad():
            acc = 0
            total = 0
            for te_data, te_label in test_loader:
                te_data, te_label = te_data.to(device), te_label.to(device)
                output = net(te_data)
                pred = torch.argmax(output, dim = 1)
                acc += (pred == te_label).sum().item() / len(te_label)
        acc = acc / len(test_loader)
        print("accuracy in test:{}%".format(acc*100))
torch.save(net.state_dict(), "net_finetuned.model")
