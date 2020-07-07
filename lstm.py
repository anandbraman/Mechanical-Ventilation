import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from sklearn.metrics import roc_curve, auc
import matplotlib as plt


torch.manual_seed(308)

input_size = 5
sequence_length = 48
num_layers = 1
hidden_size = 8
learning_rate = 0.001
batch_size = 128
num_epochs = 100


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, self.num_classes)

    def init_hidden(self, x):
        batch_size = x.size()[0]
        self.hidden_cell = (torch.zeros(batch_size, sequence_length,
                                        input_size),
                            torch.zeros(batch_size, sequence_length,
                                        input_size))

    def forward(self, x):
        # input data x
        lstm_out, self.hidden_cell = self.lstm(x.view(batch_size, len(x), -1),
                                               self.hidden_cell)
        preds = self.fc(lstm_out.view(-1, batch_size))
        return preds.view(-1)


# data goes to GPU
device = torch.device('cuda')
# avoiding some strange filepath issues related
# to gluster and kubernetes


X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')

# creating dataset
train_data = Dataset(X_train, y_train)
test_data = Dataset(X_test, y_test)

# converting to DataLoader obj
train_data = DataLoader(train_data)
test_data = DataLoader(test_data)

# initializing model
model = LSTM(input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers)

# setting optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# output and label list
prev_roc = 0
for epoch in range(num_epochs):
    for batch_n, (X, y) in enumerate(train_data):
        X = X.to(device)
        y = y.to(device)

        # zero out the optimizer gradient
        # no dependency between samples
        optimizer.zero_grad()
        model.init_hidden(X)
        y_pred = model(X)
        y_pred = torch.sigmoid(y_pred)

        batch_loss = loss(y_pred, y)
        batch_loss.backward()
        optimizer.step()

    output_lst = []
    label_lst = []

    for batch_n, (X, y) in enumerate(train_data):
        y_pred = model(X)
        y_pred = torch.sigmoid(y_pred)
        output_lst.append(y_pred.data)
        label_lst.append(y)

    output_lst = torch.cat(output_lst)
    label_lst = torch.cat(label_lst)
    pred_class = (y_pred > 0.5).float()
    acc = torch.mean((pred_class == label_lst).float()).item() * 100

    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    if roc_auc[2] > prev_roc[2]:
        prev_roc = roc_auc
        torch.save(model.state_dict(), 'results/best_model.pt')
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='dark orange', lw=lw,
                 label='ROC Curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.save('results/best_model_roc.png')

    print('Epoch {0} Accuracy: {1}'.format(epoch, acc))
    print('AUROC {}'.format(roc_auc[2]))
