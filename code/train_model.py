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

from dataset import USDataset
from network import FFNeuralNetwork

def run_epoch(model, trainloader, criterion, device, optimizer, mode):

    if mode=='train':
        model.train()
    elif mode=='val':
        model.eval()
    else:
        ValueError('Unkown mode type')

    epoch_loss = torch.zeros(size=(len(trainloader),))
    #print_loss = torch.zeros(shape=len(trainloader))
    #pbar = tqdm(trainloader, total = len(trainloader))
    pbar = tqdm(trainloader)
    for batch_idx, data in enumerate(pbar): #tqdm(enumerate(trainloader)):

        inputs = data['R'].to(device)
        labels = data['R_inv'].to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
  
        if mode=='train':
            loss.backward()
            optimizer.step()

        epoch_loss[batch_idx] = loss.detach()
        #print_loss[batch_idx] = loss
        #if (batch_idx %100==0) and (batch_idx>=100):
         #   print('current mean of losses ',torch.mean(torch.stack( print_loss)))
         #   print_loss = []
      
    return torch.mean(epoch_loss)

def evaluate(model, dataloader, criterion, device, numcl):
    model.eval()

    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs= np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
      
          inputs = data['image'].to(device)        
          outputs = model(inputs)
          labels = data['label']

          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          cpuout = outputs.to('cpu')

          concat_pred = np.vstack((concat_pred, cpuout))
          concat_labels = np.vstack((concat_labels, labels))
          fnames = fnames + data['filename']

    np.seterr(invalid='ignore')
    for c in range(numcl):
      avgprecs[c] = sklearn.metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

def traineval_model(dataloader_train, dataloader_val, model, criterion, optimizer, scheduler, 
                    num_epochs, device, filename):
    
    best_measure = 0
    best_epoch = -1

    trainlosses=[]
    vallosses=[]
    testperfs=[]

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

        #perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
        #testlosses.append(testloss)
        #testperfs.append(perfmeasure)

        #print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)

        #avgperfmeasure = np.mean(perfmeasure)
        #print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

        #if avgperfmeasure > best_measure:
        #    bestweights = model.state_dict()
        #    best_measure = avgperfmeasure
        #    torch.save(model, filename + '.pt')
            # Save values from best epoch
        #    vals = {'epoch':np.asarray(epoch), 'AP':np.asarray(perfmeasure), 'mAP':np.asarray(avgperfmeasure), 'labels':np.asarray(concat_labels), 'preds':np.asarray(concat_pred), 'fnames':np.asarray(fnames)}

    # save vals to file
    #vals['trainlosses'] = np.asarray(trainlosses); vals['testlosses'] = np.asarray(testlosses); vals['testperfs'] = np.asarray(testperfs)
    #outfile = filename + '.npz'
    #np.savez(outfile, **vals)

    return trainlosses, vallosses #, best_epoch, best_measure, bestweights, testlosses, testperfs

def plot_learning_curves(trainlosses, vallosses):

    #to cpu
    trainlosses = torch.stack(trainlosses).cpu()
    vallosses = torch.stack(vallosses).cpu()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(trainlosses, label='loss')
    ax[0].plot(vallosses, label='val')
    ax[0].legend()

    
    return

def runstuff(config, root_dir):
    # Data augmentations.
    # Scaling/Nomalization here ??
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }

    datasets={}

    datasets['train'] = USDataset(root_dir, config, trvaltest=0, transform=data_transforms['train'])
    datasets['val'] = USDataset(root_dir, config, trvaltest=0, transform=data_transforms['val'])

    #datasets['train'] = USDataset(root_dir=root_dir, images=config['images'], trvaltest=0, 
    #                              transform=data_transforms['train'], is_complex=config['is_complex'])
    #datasets['val'] = USDataset(root_dir=root_dir, images=config['images'], trvaltest=1, 
    #                            transform=data_transforms['val'], is_complex=config['is_complex'])

    # Dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=config['batchsize_train'], 
                                                       shuffle=True, num_workers=config['num_workers'], 
                                                       pin_memory=config['use_pinned_memory'])
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=config['batchsize_val'], 
                                                     shuffle=False, num_workers=config['num_workers'], 
                                                     pin_memory=config['use_pinned_memory'])

    # Device
    if True == config['use_gpu']:
        device= torch.device('cuda:0')
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

    trainlosses, vallosses = traineval_model(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['max_num_epochs'], device = device , filename='Task1')

    plot_learning_curves(trainlosses, vallosses)
    return

if __name__=='__main__':
    torch.manual_seed(0)


    config = dict()
    #config['use_gpu'] = True # change this to True for training on the cluster
    config['use_gpu'] = False # change this to True for training on the cluster
    config['lr'] = 0.001
    config['batchsize_train'] = 16
    config['batchsize_val'] = 64
    config['max_num_epochs'] = 3
    config['num_workers'] = 1
    config['scheduler_stepsize'] = 5
    config['scheduler_factor'] = 0.3
    config['images'] = ["Alpinion_L3-8_CPWC_hyperechoic_scatterers"]
    config['is_complex'] = False
    config['use_upper_triangular'] = False
    config['use_normalized_R'] = False
    config['use_pinned_memory'] = True


    #root_dir = str(pathlib.Path(__file__).parent.resolve())
    root_dir = 'C:/Users/espen/Documents/Skole/MASTER/code/'
    #root_dir = '/'
    
    runstuff(config, root_dir)

