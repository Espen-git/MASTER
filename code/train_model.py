# Train_model
from distutils.command.config import config
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
import sklearn.metrics
from typing import Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dataset import USDataset
from network import FFNeuralNetwork
from plot_loss import plot

def run_epoch(model, dataloader, criterion, device, optimizer, mode):

    if mode=='train':
        model.train()
    elif mode=='val':
        model.eval()
    else:
        ValueError('Unkown mode type')

    epoch_loss = torch.zeros(size=(len(dataloader),))
    for batch_idx, data in enumerate(tqdm(dataloader)):

        inputs = data['R'].to(device)
        labels = data['Ria'].to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)

        if mode=='train':
            loss.backward()
            optimizer.step()

        epoch_loss[batch_idx] = loss.item()
      
    return torch.mean(epoch_loss)

def traineval_model(dataloader_train, dataloader_val, model, criterion, optimizer, scheduler, 
                    num_epochs, device, filename, root_dir, config):
    
    best_val_loss = np.inf

    trainlosses=[]
    vallosses=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trainloss = run_epoch(model, dataloader_train, criterion, device, optimizer, mode='train')
        valloss= run_epoch(model, dataloader_val, criterion, device, optimizer, mode='val') 
        trainlosses.append(trainloss)
        vallosses.append(valloss)
        print(f'Epoch: {epoch} | train loss: {trainloss: 0.4f} | val loss:  {valloss: 0.4f}')

        if scheduler is not None:
            scheduler.step()

        # Check if current val loss is the so far lowest, if yes, store the model.
        if valloss <= best_val_loss:
            torch.save(model, root_dir + 'data/models/' + filename + '.pt')

        # Save values to file
        vals = {'trainlosses':np.asarray(trainlosses), 'vallosses':np.asarray(vallosses), 'config':config}
        np.savez(root_dir + 'data/models/' + filename + '.npz' , **vals)

    return trainlosses, vallosses

def plot_learning_curves(model):

    plot(model)

    return

def runstuff(config, root_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }

    datasets={}
    datasets['train'] = USDataset(root_dir, config, trvaltest=0, transform=data_transforms['train'], seed=config['seed'])
    datasets['val'] = USDataset(root_dir, config, trvaltest=1, transform=data_transforms['val'], seed=config['seed'])

    # Dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=config['batchsize_train'], 
                                                       shuffle=config['shuffle'], num_workers=config['num_workers'], 
                                                       pin_memory=config['use_pinned_memory'], prefetch_factor=config['prefetch_factor'],
                                                       persistent_workers=config['persistent_workers'])
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=config['batchsize_val'], 
                                                     shuffle=False, num_workers=config['num_workers'], 
                                                     pin_memory=config['use_pinned_memory'], prefetch_factor=config['prefetch_factor'],
                                                       persistent_workers=config['persistent_workers'])

    # Device
    if config['use_gpu']:
        device= torch.device('cuda')
    else:
        device= torch.device('cpu')

    # Model
    model = FFNeuralNetwork(config)
    model = model.to(device)

    lossfct = nn.MSELoss()

    lr = config['lr']
    someoptimizer = optim.Adam(model.parameters(), lr=lr)

    # Decay LR by a factor of config['scheduler_factor'] every config['scheduler_factor'] epochs
    somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

    trainlosses, vallosses = traineval_model(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler,
                                              num_epochs= config['max_num_epochs'], device = device , filename=config['model_name'], root_dir=root_dir, config=config)

    #plot_learning_curves(config['model_name'])
    return

if __name__=='__main__':
    torch.manual_seed(0)

    config = dict()
    config['use_gpu'] = True
    config['use_pinned_memory'] = True
    config['shuffle'] = True
    config['persistent_workers'] = False
    config['seed'] = 2023
    config['batchsize_train'] = 16
    config['batchsize_val'] = 64
    config['max_num_epochs'] = 50
    config['num_workers'] = 10
    config['prefetch_factor'] = 1
    config['lr'] = 0.001
    config['scheduler_stepsize'] = 10
    config['scheduler_factor'] = 0.5
    config['images'] = ['Alpinion_L3-8_CPWC_hyperechoic_scatterers','Alpinion_L3-8_CPWC_hypoechoic']
    config['model_name'] = 'Test5(Two images)'
    config['is_complex'] = False
    config['use_upper_triangular'] = True
    config['use_normalized'] = True

    root_dir = 'C:/Users/espen/Documents/Skole/MASTER/code/'
    
    runstuff(config, root_dir)
