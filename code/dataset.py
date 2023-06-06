import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import scipy.io as sio
from shutil import rmtree
import time
from tqdm import tqdm
import torchvision.transforms as transforms

class USDataset(Dataset):
    def __init__(self, root_dir, config, trvaltest, transform, seed):

        self.root_dir = root_dir
        self.data_dir = root_dir + "data/"

        images = config['images']
        self.is_complex = config['is_complex']
        self.use_normalized = config['use_normalized']
        self.use_upper_triangular = config['use_upper_triangular']
        self.transform = transform
        self.data_filepaths = []

        # Makes list of all R/Ria files to be used
        for image in images:     
            R_dir = self.data_dir + image + "/R/"
            for filename in os.listdir(R_dir):
                filepath = R_dir + filename
                self.data_filepaths.append(filepath)
        
        """ # pre-load all data
        self.R_data = torch.zeros(len(self.data_filepaths), 16, 16, dtype=torch.cfloat)
        self.Ria_data = torch.zeros(len(self.data_filepaths), 16, 1, dtype=torch.cfloat)

        seperator = "/"
        for idx, R_path in enumerate(tqdm(self.data_filepaths)):
            split_path = R_path.split(seperator)
            split_path[-2] = "Ria"
            Ria_path = seperator.join(split_path)

            self.R_data[idx, :, :] = self.transform(sio.loadmat(R_path)['R'])
            self.Ria_data[idx, :] = self.transform(sio.loadmat(Ria_path)['Ria']) """

        print("Loading complete")
        if trvaltest==2: # No train test split
            pass
        else:
            # Train/Test split
            data_filepaths_train, data_filepaths_test = train_test_split(self.data_filepaths, test_size=0.25, random_state=seed)

            if trvaltest==0: # use train set
                self.data_filepaths = np.array(data_filepaths_train)
            if trvaltest==1: # use val set
                self.data_filepaths = np.array(data_filepaths_test)

            """ R_train, R_test, Ria_train, Ria_test, data_filepaths_train, data_filepaths_test = train_test_split(self.R_data, self.Ria_data, self.data_filepaths, test_size=0.25, random_state=seed)

            if trvaltest==0: # use train set
                self.R_data = np.array(R_train)
                self.Ria_data = np.array(Ria_train)
                self.data_filepaths = np.array(data_filepaths_train)
            if trvaltest==1: # use val set
                self.R_data = np.array(R_test)
                self.Ria_data = np.array(Ria_test)
                self.data_filepaths = np.array(data_filepaths_test) """

    def __len__(self):
        # Number of samples
        return len(self.data_filepaths)

    def __getitem__(self, idx):
        R_path = self.data_filepaths[idx]
        seperator = "/"
        split_path = R_path.split(seperator)
        split_path[-2] = "Ria"
        Ria_path = seperator.join(split_path)

        R = sio.loadmat(R_path)['R']
        Ria = sio.loadmat(Ria_path)['Ria']

        """ R_path = self.data_filepaths[idx]
        R = self.R_data[idx]
        Ria = self.Ria_data[idx] """

        if self.use_normalized:
            scale = np.mean(np.diag(R)) # Average power
            R = R / scale
            Ria = Ria * scale
        else:
            scale = 1

        if self.transform:
            R = self.transform(R)
            Ria = self.transform(Ria)

        if self.use_upper_triangular:
            upper_triangular_idx = torch.triu_indices(R.shape[1],R.shape[2] )
            R = R[0][upper_triangular_idx[0], upper_triangular_idx[1]]
                
        if not self.is_complex:
            Ria = torch.view_as_real(Ria).flatten()
            R   = torch.view_as_real(R).flatten()

        sample = {'R': R,
                  'Ria': Ria,
                  'file_path': R_path, 
                  'scale': scale}
        
        return sample
    