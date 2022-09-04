from cProfile import label
from cgi import test
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
from model import Model
import seaborn as sns
from utils import config
import logging
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import copy
import time
logging.basicConfig(level=logging.DEBUG)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    logging.debug('DEVICE: CUDA '+torch.cuda.get_device_name(0))
else:
    logging.debug('DEVICE: CPU')

writer = SummaryWriter(config['results']['path'])

logging.debug('Loading data ...')
dataset, dataloader, classes = Dataset.create()

logging.debug('Loading model ...')
model = Model.efficientnet(len(classes))
imgs, labels = next(iter(dataloader['train']))
writer.add_graph(model, imgs)


logging.debug('Loading optimizer ...')
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=eval(config['hyper']['lr']))

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=int(config['hyper']['step']),
    gamma=float(config['hyper']['gamma'])
)


def train(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    model.to(device)
    best_wts = copy.deepcopy(model.state_dict)
    best_acc = 0.0
    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        running_corrects = 0
        phase = 'train'
        for inputs, labels in (pbar := tqdm(dataloader[phase])):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pbar.set_description(f"Epoch {epoch} loss : {running_loss:.2f} ")
        scheduler.step()
        print(running_corrects)
        training_loss = running_loss/len(dataset[phase])
        training_acc = running_corrects.double()/len(dataset[phase])
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, 'test', criterion)
        precision, recall, f1 = score(y_true, y_pred)

        print(f'Epoch {epoch} \
            train loss: {training_loss:.2f} \
            test loss: {test_loss:.2f} \
            train acc: {training_acc:.4f} \
            test acc: {test_acc:.4f} \
            precision: {precision:.2f} \
            recall: {recall:.2f} \
            F1 score: {f1:.2f} \
            ')

        writer.add_scalar('Training Loss', training_loss, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Training Acc', training_acc, epoch)
        writer.add_scalar('Test Acc', test_acc, epoch)
        writer.add_scalar('Precision', precision, epoch)
        writer.add_scalar('Recall', recall, epoch)
        writer.add_scalar('F1 score', f1, epoch)

        if test_acc >= best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_wts)
    test_loss, test_acc, y_true, y_pred = evaluate(model, 'test', criterion)
    report(y_true, y_pred)
    return model


def score(labels, preds):
    precision = precision_score(labels, preds, average='micro')
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='micro')
    return precision, recall, f1


def report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(config['results']['path']+'classification_report.csv')
    print(report)


def cm_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True)
    plt.savefig(config['results']['path']+'confusion_matrix.png')


def evaluate(model, phase, criterion):
    model.eval()
    preds = torch.Tensor().to(device)
    labels = torch.Tensor().to(device)
    running_loss = 0.0
    running_corrects = 0
    for x, y_true in (pbar := tqdm(dataloader[phase])):
        x = x.to(device)
        y_true = y_true.to(device)
        labels = torch.cat((labels, y_true.clone().detach()))
        with torch.no_grad():
            outputs = model(x)
            _, y_pred = torch.max(outputs, 1)
            loss = criterion(outputs, y_pred)
            running_loss += loss.item()*x.size(0)
            running_corrects += torch.sum(y_pred == y_true.data)
            preds = torch.cat((preds, y_pred.clone().detach()))
    print(running_corrects)
    loss = running_loss/len(dataset[phase])
    acc = running_corrects.double()/len(dataset[phase])
    labels = np.array(labels.cpu())
    preds = np.array(preds.cpu())
    return loss, acc, labels, preds


model = train(model, criterion, optimizer, scheduler, 25)
torch.save(model.state_dict(), config['results']['path']+'best.pt')
writer.close()
