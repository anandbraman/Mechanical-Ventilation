import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, precision_recall_curve, recall_score, f1_score
import matplotlib.pyplot as plt
# from experiment_tracking import ExperimentTracker
import csv


torch.manual_seed(308)

input_size = 5
sequence_length = 48
num_layers = 1
hidden_size = 8
lr = 0.001
num_epochs = 500
# data and model go to GPU
device = torch.device('cuda')

# epochs, hiddensize, LR
model_id = "GRU_" + str(num_epochs) + '_' + str(hidden_size) + \
    '_' + str(lr).split('.')[1]

# experiment_tracker = ExperimentTracker(
#    'client_secret.json', 'experiment-tracking')

# experiment_tracker.unique_params(model_id, 'model_id')

# creating a model_id folder in which plots can be saved
if not os.path.isdir('results/' + model_id):
    os.mkdir('results/' + model_id)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = 2
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, self.num_classes)

    def init_hidden(self, x):
        self.batch_size = x.size()[0]
        self.hidden_cell = torch.zeros(num_layers, self.batch_size,
                                        hidden_size, device=device)

    def forward(self, x):
        # input data x
        # for the view call: batch size, sequence length, cols
        gru_out, self.hidden_cell = self.gru(x.view(self.batch_size,
                                                      x.size()[1], -1),
                                               self.hidden_cell)

        preds = self.fc(gru_out.reshape(self.batch_size, -1))
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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()


prev_roc = 0
loss_lst = []
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
    y_pred_lst = []
    label_lst = []

    for batch_n, (X, y) in enumerate(train_data):
        X = X.float().to(device)
        # y as a float this time for bc no call to nn.CrossEntropyLoss
        y = y.float().to(device)
        model.init_hidden(X)
        y_pred = model(X)
        # append prediction probabilities for classes 0 and 1
        # used in loss computation
        y_pred_lst.append(y_pred)
        # pulling out the prediction of class 1
        y_pos_pred = y_pred[:, 1]
        # turning into probability
        y_pos_pred = torch.sigmoid(y_pos_pred).cuda()
        output_lst.append(y_pos_pred.data)
        label_lst.append(y)

    output = torch.cat(output_lst)
    label = torch.cat(label_lst)
    y_pred_df = torch.cat(y_pred_lst)
    pred_class = (output > 0.5).float()
    acc = torch.mean((pred_class == label).float()).item() * 100
    # label must be a long in crossentropy loss calc
    epoch_loss = loss(y_pred_df, label.long().view(-1))
    # must graph the epoch losses to check for convergence
    loss_lst.append(epoch_loss.item())

    # must put tensors on the cpu to convert to numpy array
    fpr, tpr, _ = roc_curve(label.cpu(), output.cpu())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    if roc_auc > prev_roc:
        prev_roc = roc_auc
        fpath = 'results/' + model_id + '/' + model_id + '.pt'
        torch.save(model.state_dict(), fpath)
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
        plot_path = 'results/' + model_id + '/' + model_id + '_train_roc.png'
        plt.savefig(plot_path)
        plt.close()

    print('Epoch {0} Accuracy: {1}'.format(epoch, acc))
    print('AUROC {}'.format(roc_auc))

# checking for convergence by plotting loss after each epoch
plt.figure()
lw = 2
plt.plot(range(num_epochs), loss_lst)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Convergence')
loss_plot_path = 'results/' + model_id + '/' + model_id + '_loss.png'
plt.savefig(loss_plot_path)
plt.close()

# test results
best_model = LSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers)

best_model_path = 'results/' + model_id + '/' + model_id + '.pt'
best_model_sd = torch.load(best_model_path)
best_model.load_state_dict(best_model_sd)
best_model.to(device)

test_output_lst = []
test_label_lst = []
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
    test_output_lst.append(y_pred.data)
    test_label_lst.append(y)

test_output = torch.cat(test_output_lst)
test_label = torch.cat(test_label_lst)
test_class = (test_output > 0.5).float()
test_acc = torch.mean((test_class == test_label).float()).item() * 100
test_precision = precision_score(test_label.cpu(), test_class.cpu())
test_recall = recall_score(test_label.cpu(), test_class.cpu())
test_f1 = f1_score(test_label.cpu(), test_class.cpu())

precision, recall, _ = precision_recall_curve(test_label.cpu(),
                                              test_output.cpu())
# precision recall plot
plt.figure()
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='upper right')
plt.title('Test Precision Recall Curve')
pr_path = 'results/' + model_id + '/' + model_id + '_precision_recall.png'
plt.savefig(pr_path)
plt.close()

# must put tensors on the cpu to convert to numpy array
# plotting the ROC curve as well
test_fpr, test_tpr, _ = roc_curve(test_label.cpu(), test_output.cpu())
test_roc_auc = auc(test_fpr, test_tpr)

plt.figure()
lw = 2
plt.plot(test_fpr, test_tpr, color='darkorange', lw=lw,
         label='ROC Curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mechanical Ventilation Test ROC')
plt.legend(loc="lower right")
test_roc_path = 'results/' + model_id + '/' + model_id + '_test_roc.png'
plt.savefig(test_roc_path)
plt.close()

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
