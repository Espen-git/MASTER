# Dataset
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os

torch.manual_seed(0)

class USDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):

        self.root_dir = root_dir
        self.data_dir = root_dir + "data/"
        self.transform = transform
        self.data_filenames=[]

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".npy"):
                # Gets only .npy files present in data folder
                self.data_filenames.append(filename)
  
        # Train/Test split
        self.data_filenames_train, self.data_filenames_test = train_test_split(self.data_filenames, test_size=0.33, random_state=0)

        if trvaltest==0: # train
            self.data_filenames = self.data_filenames_train
        if trvaltest==1: # val
            self.data_filenames = self.data_filenames_test

    def __len__(self):
        # Number of samples
        return len(self.data_filenames)

    def __getitem__(self, idx):
        both_R = np.load(self.data_dir + self.data_filenames[idx], mmap_mode='r')
        R = both_R[:,:,0]
        R_inv = both_R[:,:,1]

        if self.transform:
            R = self.transform(R)
            R_inv = self.transform(R_inv)

        # trekant
        sample = {'R': R,
                  'R_inv': R_inv,
                  'filename': self.data_filenames[idx]}
        
        return sample