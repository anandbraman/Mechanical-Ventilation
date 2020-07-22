import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# from experiment_tracking import ExperimentTracker
import csv
# file with useful modeling functions
import nnfuncs
from sklearn.metrics import auc, roc_curve
import csv_tracker as track

torch.manual_seed(308)

# num obs in training sequence
sequence_length = 48
num_layers = 1
hidden_size = 8
lr = 0.001
num_epochs = 500
# data and model go to GPU
device = torch.device('cuda')

# epochs, hiddensize, LR
model_id = "LSTM_" + str(num_epochs) + '_' + str(hidden_size) + \
    '_' + str(lr).split('.')[1]

# experiment_tracker = ExperimentTracker(
#    'client_secret.json', 'experiment-tracking')

# experiment_tracker.unique_params(model_id, 'model_id')

# creating a model_id folder in which plots can be saved
if not os.path.isdir('results/' + model_id):
    os.mkdir('results/' + model_id)


class LSTM(nn.Module):

    def __init__(self, sequence_length, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        # 5 predictors
        self.input_size = 5
        # binary classification
        self.num_classes = 2
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size,
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


# reading in the data
X_train = torch.load('data/X_train.pt')
y_train = torch.load('data/y_train.pt')
X_val = torch.load('data/X_val.pt')
y_val = torch.load('data/y_val.pt')
# concatenating train and val for full dataset
X_train_full = torch.cat((X_train, X_val))
y_train_full = torch.cat((y_train, y_val))
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')

# creating dataset
train_data = nnfuncs.build_dataset(X_train, y_train, 128)
val_data = nnfuncs.build_dataset(X_val, y_val, 128)
full_train_data = nnfuncs.build_dataset(X_train_full, y_train_full, 128)
test_data = nnfuncs.build_dataset(X_test, y_test, 128)


# initializing model
model = LSTM(sequence_length=sequence_length, hidden_size=hidden_size,
             num_layers=num_layers)

# send to gpu
model.cuda()

# setting optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

# initializing roc value and list to track train and val losses after each epoch
prev_roc = 0
train_loss = []
val_loss = []
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

    # training and validation outputs, predictions of class 1, labels
    train_output, train_pred, train_label = nnfuncs.get_preds_labels(model,
                                                                     train_data,
                                                                     device=device)
    val_output, val_pred, val_label = nnfuncs.get_preds_labels(model,
                                                               val_data,
                                                               device=device)

    # calculating accuracy at 0.5 threshold
    acc = nnfuncs.model_accuracy(val_pred, val_label, 0.5)
    # label must be a long in crossentropy loss calc
    epoch_loss_train = loss(train_output, train_label.long().view(-1))
    epoch_loss_val = loss(val_output, val_label.long().view(-1))

    # must graph the epoch losses to check for convergence
    # appending loss vals to list after each epoch
    train_loss.append(epoch_loss_train.item())
    val_loss.append(epoch_loss_val.item())

    # must put tensors on the cpu to convert to numpy array
    # getting validation set roc and saving model
    # if roc improves
    fpr, tpr, _ = roc_curve(val_label.cpu(), val_pred.cpu())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    if roc_auc > prev_roc:
        # save over roc
        prev_roc = roc_auc
        roc_path = 'results/' + model_id + '/' + model_id + '_val_roc.png'
        nnfuncs.plot_roc(fpr, tpr, roc_auc, roc_path)
        model_path = 'results/' + model_id + '/' + model_id + '.pt'
        torch.save(model.state_dict(), model_path)

    print('Epoch {0} Validation Set Accuracy: {1}'.format(epoch, acc))
    print('Validation Set AUROC {}'.format(roc_auc))


nnfuncs.plot_loss(model_id, train_loss, val_loss)

# test results
best_model = LSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers)

best_model_path = 'results/' + model_id + '/' + model_id + '.pt'
best_model_sd = torch.load(best_model_path)
best_model.load_state_dict(best_model_sd)
best_model.to(device)

test_output, test_pred, test_label = nnfuncs.get_preds_labels(best_model,
                                                              test_data,
                                                              device)


test_acc = nnfuncs.model_accuracy(test_pred, test_label, 0.5)
test_precision, test_recall, test_f1 = nnfuncs.precision_recall_f1(test_label,
                                                                   test_pred,
                                                                   0.5)

pr_path = 'results/' + model_id + '/' + model_id + '_precision_recall.png'
nnfuncs.plot_test_precision_recall(pr_path, test_label, test_pred)


# plotting and saving ROC curve as well
test_fpr, test_tpr, _ = roc_curve(test_label.cpu(), test_output.cpu())
test_roc_auc = auc(test_fpr, test_tpr)
test_roc_path = 'results/' + model_id + '/' + model_id + '_test_roc.png'
nnfuncs.plot_roc(test_fpr, test_tpr, test_roc_auc, test_roc_path)

experiment_header = ['model_type', 'model_id', 'epochs',
                     'learning_rate', 'hidden_size', 'loss', 'precision',
                     'recall', 'f1_score', 'auroc', 'accuracy']

experiment_params = ['LSTM', model_id, num_epochs, lr,
                     hidden_size, loss_lst[-1], test_precision,
                     test_recall, test_f1, test_roc_auc, test_acc]

# writing results to a csv file
if not os.path.isfile('results/experiment_tracking.csv'):
    with open('results/experiment_tracking.csv', 'a+', newline='') as f:
        Writer = csv.writer(f)
        Writer.writerow(experiment_header)
        Writer.writerow(experiment_params)
else:
    with open('results/experiment_tracking.csv', 'a+', newline='') as f:
        Writer = csv.writer(f)
        Writer.writerow(experiment_params)

# experiment_tracker.record_experiment(experiment_params)
