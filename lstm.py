import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


torch.manual_seed(308)

input_size = 5
sequence_length = 48
num_layers = 1
hidden_size = 8
learning_rate = 0.001
num_epochs = 100
# data and model go to GPU
device = torch.device('cuda')


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = 2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, self.num_classes)

    def init_hidden(self, x):
        self.batch_size = x.size()[0]
        self.hidden_cell = (torch.zeros(num_layers, self.batch_size,
                                        hidden_size, device=device),
                            torch.zeros(num_layers, self.batch_size,
                                        hidden_size, device=device))

    def forward(self, x):
        # input data x
        # for the view call: batch size, sequence length, cols
        lstm_out, self.hidden_cell = self.lstm(x.view(self.batch_size,
                                                      x.size()[1], -1),
                                               self.hidden_cell)

        preds = self.fc(lstm_out.reshape(self.batch_size, -1))
        return preds.view(self.batch_size, -1)


X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')

# creating dataset
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# converting to DataLoader obj
train_data = DataLoader(train_data, batch_size=128)
test_data = DataLoader(test_data, batch_size=128)

# initializing model
model = LSTM(input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers)

# send to gpu
model.cuda()

# setting optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# output and label list
# roc output is of len 3
prev_roc = 0
for epoch in range(num_epochs):
    for batch_n, (X, y) in enumerate(train_data):
        X = X.float().to(device)
        # cross entropy loss takes an int as the target which corresponds
        # to the index of the target class
        # output should be of shape (batch_size, num_classes)
        y = y.long().to(device)

        # zero out the optimizer gradient
        # no dependency between samples
        optimizer.zero_grad()
        model.init_hidden(X)
        y_pred = model(X)
        # print(y_pred.size())
        # print(y.size())
        batch_loss = loss(y_pred, y.view(-1))
        batch_loss.backward()
        optimizer.step()

    output_lst = []
    label_lst = []

    for batch_n, (X, y) in enumerate(train_data):
        X = X.float().to(device)
        # y as a float this time for bc no call to nn.CrossEntropyLoss
        y = y.float().to(device)
        model.init_hidden(X)
        y_pred = model(X)
        y_pred = y_pred[:, 1]
        y_pred = torch.sigmoid(y_pred).cuda()
        output_lst.append(y_pred.data)
        label_lst.append(y)

    output = torch.cat(output_lst)
    label = torch.cat(label_lst)
    pred_class = (output > 0.5).float()
    acc = torch.mean((pred_class == label).float()).item() * 100

    # must put tensors on the cpu to convert to numpy array
    fpr, tpr, _ = roc_curve(label.cpu(), output.cpu())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    if roc_auc > prev_roc:
        prev_roc = roc_auc
        torch.save(model.state_dict(), 'results/best_model.pt')
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
                 label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mechanical Ventilation Train ROC')
        plt.legend(loc="lower right")
        plt.savefig('results/best_model_train_roc.png')

    print('Epoch {0} Accuracy: {1}'.format(epoch, acc))
    print('AUROC {}'.format(roc_auc))

# test results
best_model = LSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers)

best_model_sd = torch.load('results/best_model.pt')
best_model.load_state_dict(best_model_sd)
best_model.to(device)

# iterating over the test data
for batch_n, (X, y) in enumerate(test_data):
    X = X.float().to(device)
    # y as a float this time for bc no call to nn.CrossEntropyLoss
    y = y.float().to(device)
    model.init_hidden(X)
    y_pred = model(X)
    # select prediction of class 1
    y_pred = y_pred[:, 1]
    y_pred = torch.sigmoid(y_pred).cuda()
    output_lst.append(y_pred.data)
    label_lst.append(y)

output = torch.cat(output_lst)
label = torch.cat(label_lst)
pred_class = (output > 0.5).float()
acc = torch.mean((pred_class == label).float()).item() * 100

# must put tensors on the cpu to convert to numpy array
fpr, tpr, _ = roc_curve(label.cpu(), output.cpu())
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw,
         label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mechanical Ventilation Test ROC')
plt.legend(loc="lower right")
plt.savefig('results/best_model_test_roc.png')
