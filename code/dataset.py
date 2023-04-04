# Dataset
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import scipy.io as sio
from shutil import rmtree

torch.manual_seed(0)

class USDataset(Dataset):
    def __init__(self, root_dir, images, trvaltest, transform):

        self.root_dir = root_dir
        self.data_dir = root_dir + "data/"

        self.transform = transform
        self.data_filepaths=[]

        # Makes list of all R/Rinv files to be used, and saves copy of R and Rinv to tmp directory
        for image in images:
            R_dir = self.data_dir + image + "/R/"
            Ria_dir = self.data_dir + image + "/Ria/"
            for filename in os.listdir(R_dir):
                filepath = R_dir + filename
                self.data_filepaths.append(filepath)

        # Train/Test split
        self.data_filepaths_train, self.data_filepaths_test = train_test_split(self.data_filepaths, test_size=0.25, random_state=0)

        print("Test/Train split complete")

        if trvaltest==0: # use train set
            self.data_filepaths = self.data_filepaths_train
        if trvaltest==1: # use val set
            self.data_filepaths = self.data_filepaths_test

    def __len__(self):
        # Number of samples
        return len(self.data_filepaths)

    def __getitem__(self, idx):
        # Add flatten
        R_path = self.data_filepaths[idx]
        seperator = "/"
        split_path = R_path.split(seperator)
        split_path[-2] = "Ria"
        filename = split_path[-1]
        Ria_path = seperator.join(split_path)

        R = np.load(R_path, mmap_mode='r')
        Ria = np.load(Ria_path, mmap_mode='r')

        if self.transform:
            R = self.transform(R)
            Ria = self.transform(Ria)

        # trekant?
        sample = {'R': R,
                  'R_inv': Ria,
                  'filename': filename}
        
        return sample