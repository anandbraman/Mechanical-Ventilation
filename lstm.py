import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.manual_seed(308)

input_size = 5
sequence_length = 48
num_layers = 1
hidden_size = 8
learning_rate = 0.001
batch_size = 128
num_epochs = 2


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

X_train = torch.load('~/files/X_train.pt')
y_train = torch.load('~/files/y_train.pt')
X_test = torch.load('~/files/X_test.pt')
y_test = torch.load('~/files/y_test.pt')

train_data = DataLoader(X_train, y_train)
test_data = DataLoader(X_test, y_test)

# initializing model
model = LSTM(input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers)

# setting optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_n, (X, y) in enumerate(train_data):
        X = X.to(device)
        y = y.to(device)

        # zero out the optimizer gradient
        # no dependency between samples
        optimizer.zero_grad()
        model.init_hidden(X)
        y_pred = model(X)

        batch_loss = loss(y_pred, y)
        batch_loss.backward()
        optimizer.step()
