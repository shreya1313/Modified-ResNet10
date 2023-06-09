import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import torch.utils.data as data
import torchsummary
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd

from models.resnet10 import ResNet10 
from models.resnet18 import ResNet18
from models.resnet12 import ResNet12
from models.resnet14_4 import ResNet14_4
from models.resnet14_5 import ResNet14_5
from models.resnet10 import BasicBlock as BasicBlock10
from models.resnet18 import BasicBlock as BasicBlock18
from models.resnet12 import BasicBlock

# Compute means and standard deviations along the R,G,B channel
print("Download start")
ROOT = "./CIFAR10/"
train_data = datasets.CIFAR10(root = ROOT, 
                              train = True, 
                              download = True)
print("Download end")
means = train_data.data.mean(axis = (0,1,2)) / 255
stds = train_data.data.std(axis = (0,1,2)) / 255
print(means)
print(stds)
train_transforms = transforms.Compose([
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(32, padding = 2),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, 
                                                std = stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, 
                                                std = stds)
                       ])


train_data = datasets.CIFAR10(ROOT, 
                              train = True, 
                              download = True, 
                              transform = train_transforms)

test_data = datasets.CIFAR10(ROOT, 
                             train = False, 
                             download = True, 
                             transform = test_transforms)

VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

BATCH_SIZE = 256

train_iterator = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
valid_iterator = torch.utils.data.DataLoader(valid_data,batch_size=BATCH_SIZE,shuffle=False)
test_iterator = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


def calculate_accuracy(y_pred, y):
    """
    Calculates the accuracy of predictions given a predicted output tensor and ground truth tensor.

    Args:
        y_pred (torch.Tensor): The predicted output tensor, with shape (batch_size, num_classes).
        y (torch.Tensor): The ground truth tensor, with shape (batch_size,).

    Returns:
        float: The accuracy of the predictions as a float value between 0 and 1.

    """
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    """
    Trains a PyTorch model for one epoch given an iterator of data, an optimizer, a loss criterion, and a device.

    Args:
        model (nn.Module): The PyTorch model to train.
        iterator (DataLoader): The iterator of data to use for training.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss criterion to use for training.
        device (str): The device to use for training, either 'cpu' or 'cuda'.

    Returns:
        tuple: A tuple of floats representing the average epoch loss and epoch accuracy, respectively.

    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)
        
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluates a PyTorch model on a given iterator of data using a loss criterion and device.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        iterator (DataLoader): The iterator of data to use for evaluation.
        criterion (nn.Module): The loss criterion to use for evaluation.
        device (str): The device to use for evaluation, either 'cpu' or 'cuda'.

    Returns:
        tuple: A tuple of floats representing the average epoch loss and epoch accuracy, respectively.

    """
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def update_lr(optimizer, lr): 
    """
    Updates the learning rate of a given PyTorch optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to update.
        lr (float): The new learning rate to set for the optimizer.

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_to_excel(data, table_name):
    """
    Saves a dictionary of data to an Excel file with the given table name.

    Args:
        data (dict): The dictionary of data to save to an Excel file.
        table_name (str): The name to use for the Excel table.

    """
  df = pd.DataFrame(data=data)
  df = df.T

  df.to_excel(f'./results/{table_name}.xlsx')



def run_exp(model, optimizer, learning_rate, curr_lr, table_name = '', EPOCHS = 50):
    """
    Runs a training and evaluation loop for a given PyTorch model, using the specified optimizer and learning rate.

    Args:
        model (torch.nn.Module): The PyTorch model to train and evaluate.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer to use for training.
        learning_rate (float): The initial learning rate to use for the optimizer.
        curr_lr (float): The current learning rate being used for the optimizer.
        table_name (str, optional): The name to use for the results table in the Excel file. Defaults to an empty string.
        EPOCHS (int, optional): The number of epochs to train the model for. Defaults to 50.

    Returns:
        tuple: A tuple containing the training losses, validation losses, and validation accuracies as lists.

    """
  max_validation_accuracy = 0
  train_losses = []
  valid_losses = []
  valid_accuracies = []

  for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
    print(f'Epoch {epoch}, Train loss {train_loss}, Train accuracy {train_acc}, Validation loss {valid_loss}, Validation accuracy {valid_acc}')

    if(valid_acc >= max_validation_accuracy):
      print(f"Validation accuracy increased from {max_validation_accuracy:.2f} to {valid_acc:.2f}")
      max_validation_accuracy = valid_acc
      torch.save(model.state_dict(), './weights/' + table_name +'.ckpt')

    if (epoch+1) % 20 == 0:
          curr_lr /= 3
          update_lr(optimizer, curr_lr)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)

  save_to_excel([train_losses,valid_losses,valid_accuracies], table_name)

  return train_losses, valid_losses, valid_accuracies




def run():
    """
    Train and evaluate ResNet models with different hyperparameters.

    For each combination of learning rate and optimizer, this function trains ResNet10, ResNet18, ResNet12, ResNet14_4, and
    ResNet14_5 models and saves the weights to a file named 'resnet<depth>_<optimizer>_<learning_rate>.ckpt', where <depth>
    is the depth of the ResNet model. Then, it evaluates the models on the test set and saves the test loss and accuracy
    to an Excel file named 'resnet<depth>_<optimizer>_<learning_rate>_acc.xlsx'. Finally, it prints the test loss,
    test accuracy, and the name of the model that was evaluated.

    """
  learning_rate = [0.1, 0.01, 0.001]
  optimizers = ['Adam', 'SGD', 'AdaDelta']
  for lr in learning_rate:
    for optimizer in optimizers:
      resnet10 = ResNet10(BasicBlock10, [2, 1, 1, 1]).to(device)
      resnet18 = ResNet18(BasicBlock18, [2, 2, 2, 2]).to(device)
      resnet12 = ResNet12(BasicBlock, [2, 2, 2]).to(device)
      resnet14_4 = ResNet14_4(BasicBlock, [2, 2, 2, 1]).to(device)
      resnet14_5 = ResNet14_5(BasicBlock, [2, 3, 3, 1, 1]).to(device)

      if optimizer == 'Adam':
        opt_10 = optim.Adam(resnet10.parameters(), lr = lr)
        opt_12 = optim.Adam(resnet12.parameters(), lr = lr)
        opt_14_4 = optim.Adam(resnet14_4.parameters(), lr = lr)
        opt_14_5 = optim.Adam(resnet14_5.parameters(), lr = lr)
        opt_18 = optim.Adam(resnet18.parameters(), lr = lr)
      elif optimizer == 'SGD':
        opt_10 = optim.SGD(resnet10.parameters(), lr = lr, weight_decay= 0.0001, momentum = 0.9)
        opt_12 = optim.SGD(resnet12.parameters(), lr = lr, weight_decay= 0.0001, momentum = 0.9)
        opt_14_4 = optim.SGD(resnet14_4.parameters(), lr = lr, weight_decay= 0.0001, momentum = 0.9)
        opt_14_5 = optim.SGD(resnet14_5.parameters(), lr = lr, weight_decay= 0.0001, momentum = 0.9)
        opt_18 = optim.SGD(resnet18.parameters(), lr = lr, weight_decay= 0.0001, momentum = 0.9)
      elif optimizer == 'AdaDelta':
        opt_10 = optim.Adadelta(resnet10.parameters(), lr=lr, weight_decay=0.0001)
        opt_12 = optim.Adadelta(resnet12.parameters(), lr=lr, weight_decay=0.0001)
        opt_14_4 = optim.Adadelta(resnet14_4.parameters(), lr=lr, weight_decay=0.0001)
        opt_14_5 = optim.Adadelta(resnet14_5.parameters(), lr=lr, weight_decay=0.0001)
        opt_18 = optim.Adadelta(resnet18.parameters(), lr=lr, weight_decay=0.0001)
       
      run_exp(resnet10, opt_10, lr, lr, 'resnet10_'+optimizer+'_'+str(lr), EPOCHS = 70)
      run_exp(resnet18, opt_18, lr, lr, 'resnet18_'+optimizer+'_'+str(lr), EPOCHS = 70)
      run_exp(resnet12, opt_12, lr, lr, 'resnet12_'+optimizer+'_'+str(lr), EPOCHS = 70)
      run_exp(resnet14_4, opt_14_4, lr, lr, 'resnet14_4_'+optimizer+'_'+str(lr), EPOCHS = 70)
      run_exp(resnet14_5, opt_14_5, lr, lr, 'resnet14_5_'+optimizer+'_'+str(lr), EPOCHS = 70)

      resnet10.load_state_dict(torch.load('./weights/' + 'resnet10_'+optimizer+'_'+str(lr) +'.ckpt'))
      test_loss, test_acc = evaluate(resnet10, test_iterator, criterion, device)
      save_to_excel([test_loss,test_acc], 'resnet10_'+optimizer+'_'+str(lr)+'_acc')
      print('Test loss: ' + str(test_loss), 'Test accuracy: ' + str(test_acc), ' resnet10_'+optimizer+'_'+str(lr))
        
        
      resnet18.load_state_dict(torch.load('./weights/' + 'resnet18_'+optimizer+'_'+str(lr) +'.ckpt'))
      test_loss, test_acc = evaluate(resnet18, test_iterator, criterion, device)
      save_to_excel([test_loss,test_acc], 'resnet18_'+optimizer+'_'+str(lr)+'_acc')
      print('Test loss: ' + str(test_loss), 'Test accuracy: ' + str(test_acc), ' resnet18_'+optimizer+'_'+str(lr))
        
        
      resnet12.load_state_dict(torch.load('./weights/' + 'resnet12_'+optimizer+'_'+str(lr) +'.ckpt'))
      test_loss, test_acc = evaluate(resnet12, test_iterator, criterion, device)
      save_to_excel([test_loss,test_acc], 'resnet12_'+optimizer+'_'+str(lr)+'_acc')
      print('Test loss: ' + str(test_loss), 'Test accuracy: ' + str(test_acc), ' resnet12_'+optimizer+'_'+str(lr))
        
        
      resnet14_4.load_state_dict(torch.load('./weights/' + 'resnet14_4_'+optimizer+'_'+str(lr) +'.ckpt'))
      test_loss, test_acc = evaluate(resnet14_4, test_iterator, criterion, device)
      save_to_excel([test_loss,test_acc], 'resnet14_4_'+optimizer+'_'+str(lr)+'_acc')
      print('Test loss: ' + str(test_loss), 'Test accuracy: ' + str(test_acc), ' resnet14_4_'+optimizer+'_'+str(lr))
    
      resnet14_5.load_state_dict(torch.load('./weights/' + 'resnet14_5_'+optimizer+'_'+str(lr) +'.ckpt'))
      test_loss, test_acc = evaluate(resnet14_5, test_iterator, criterion, device)
      save_to_excel([test_loss,test_acc], 'resnet14_5_'+optimizer+'_'+str(lr)+'_acc')
      print('Test loss: ' + str(test_loss), 'Test accuracy: ' + str(test_acc), ' resnet14_5_'+optimizer+'_'+str(lr))
    
      

run()
