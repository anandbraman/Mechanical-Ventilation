import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as sm


def build_dataset(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    tensordata = TensorDataset(X, y)
    dataloader = DataLoader(tensordata, batch_size=batch_size)
    return dataloader


def get_preds_labels(model, dataloader, device=torch.device('cuda')):
    '''
    Iterates over a torch dataloader object and returns model output, prediction probabily for class 1
    and a list of labels

    torch model, torch dataloader, optional device -> tensor, tensor, tensor

    Inputs:
        torch model, optimizer, train or test data, device if wanted

    Outputs:
        model output, predictions probabilities for class 1, labels

    '''

    # output of model(data)
    # appends batches
    output_lst = []
    # labels for each batch
    label_lst = []

    for batch_n, (X, y) in enumerate(dataloader):
        X = X.float().to(device)
        y = y.float().to(device)
        model.init_hidden()
        output = model(X)
        output = output.data
        output_lst.append(output)
        label_lst.append(y)

    # output is n x num_classes
    output_df = torch.cat(output_lst)
    y_pred = output_df[:, 1]
    y_pred = torch.sigmoid(y_pred)
    label_df = torch.cat(label_lst)

    return output_df, y_pred, label_df


def model_accuracy(y_pred: torch.Tensor, label: torch.Tensor, thresh: float):
    '''
    Given probabilities and true labels, calculates accuracy based on threshold

    tensor, tensor, float -> float

    Input:
        prediction probabilities for class 1, labels, threshold

    Output:
        accuracy between 0 and 100
    '''
    pred_class = (y_pred > thresh).float()
    acc = torch.mean((pred_class == label).float()).item() * 100
    return acc


def plot_roc(fpr, tpr, roc_auc, fpath):
    '''
    plot and save an roc curve using output of sklearn.metrics roc_curve

    Inputs:
        false pos rate vector, true pos rate vector, AUROC, filepath

    Outputs:
        None
    '''
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
    plt.savefig(fpath)
    plt.close()


def plot_loss(model_id, train_loss, val_loss):
    '''
    Plot the loss values by epoch of the train set and validation set
    and save the result


    Inputs:
        List of train loss vals, list of validation loss vals


    Output:
        none
    '''

    plt.figure()
    plt.plot(range(len(train_loss)), train_loss,
             color='navy', label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss,
             color='darkorange', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='top right')
    fpath = 'results/' + model_id + '/' + model_id + '_loss.png'
    plt.savefig(fpath)
    plt.close()


def plot_test_precision_recall(fpath, label: torch.Tensor, y_pred: torch.Tensor):
    '''
    Input:
        filepath, labels, prediciton probabilities

    Output:
        None

    saves a precision recall plot to the specified filepath generated with given
    label and the predicion probabilities
    '''

    precision, recall, _ = sm.precision_recall_curve(label, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision Recall Curve')
    plt.savefig(fpath)
    plt.close()


def precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float):
    '''
    outputs precision, recall and f1 given a decision threshold

    Tensor, Tensor, float -> float, float, float

    Inputs:
        true values or label, prediction probabilities, decision threshold [0, 1]

    Outputs:
        precision, recall, f1-score at given threshold
    '''
    pred_class = (y_pred > threshold).float()
    precision = sm.precision_score(y_true, pred_class)
    recall = sm.recall_score(y_true, pred_class)
    f1 = sm.f1_score(y_true, pred_class)

    return precision, recall, f1
