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
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import os
import scipy.io as sio

from dataset import USDataset
from network import FFNeuralNetwork, FFNeuralNetwork2, FFNeuralNetwork3

def load_model_config(root_dir, modelname):
    model_path = root_dir + 'data/models/' + modelname + '.pt'
    model = torch.load(model_path)
    
    data_path = root_dir + 'data/models/' + modelname + '.npz'
    data = np.load(data_path, allow_pickle=True) 
    config = data['config'][()]
    config['batchsize'] = 16
    config['num_workers'] = 10
    config['prefetch_factor'] = 1

    return model, config

def run_forward(root_dir, model, config):
    data_transform = transforms.Compose([transforms.ToTensor()])

    dataset = USDataset(root_dir, config, trvaltest=2, transform=data_transform, seed=config['seed'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batchsize'], 
                                                     shuffle=False, num_workers=config['num_workers'], 
                                                     pin_memory=config['use_pinned_memory'])#, prefetch_factor=config['prefetch_factor'],
                                                       #persistent_workers=config['persistent_workers'])

    if config['use_gpu']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.eval()
    save_dir_exists = False
    for batch_idx, data in enumerate(tqdm(dataloader)):
        inputs = data['R'].to(device)
        Ria = model(inputs)
        batch_size = len(data['file_path'])

        if not config['is_complex']:
            Ria = torch.reshape(Ria, (batch_size, int( Ria.shape[1] / 2 ), 2))
            Ria = torch.view_as_complex(Ria)

        scale = data['scale']
        file_paths = data['file_path']

        for i in range(batch_size):
            current_Ria = Ria[i] / scale[i] 
            Ria_dict = {'Ria': np.array(current_Ria.detach().cpu())}
            # Create path
            data_path = str(file_paths[i])
            seperator = "/"
            split_path = data_path.split(seperator)

            if not save_dir_exists:
                # Create save directory
                image_name = split_path[-3]
                save_dir = 'Ria_' + config['model_name']
                save_dir_path = root_dir + 'data/' + image_name + '/' + save_dir
                if not os.path.exists(save_dir_path):
                    os.mkdir(save_dir_path)
                save_dir_exists = True

            file_name = split_path[-1]
            file_path = save_dir_path + '/' + file_name
            
            sio.savemat(file_path, Ria_dict)
            #tmp = sio.loadmat(file_path)

if __name__ == '__main__':

    root_dir = 'C:/Users/espen/Documents/Skole/MASTER/code/'
    modelname = 'Test12(Verasonics-small_network_4_frames)'

    model, config = load_model_config(root_dir, modelname)
    config['images'] = ['Verasonics_P2-4_parasternal_long_small_1_frame']
    config['model_name'] = modelname

    run_forward(root_dir, model, config)
