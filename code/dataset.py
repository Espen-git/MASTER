# Dataset
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import scipy.io as sio

torch.manual_seed(0)

class USDataset(Dataset):
    def __init__(self, root_dir, images, trvaltest, transform):

        self.root_dir = root_dir
        self.data_dir = root_dir + "data/"
        self.tmp_dir_R = self.data_dir + "tmp/R/"
        self.tmp_dir_Ria = self.data_dir + "tmp/Ria/"
        self.transform = transform
        self.data_filenames=[]

        # Makes list of all R/Rinv files to be used, and saves copy of R and Rinv to tmp directory
        for image in images:
            R_dir = self.data_dir + image + "/R/"
            Ria_dir = self.data_dir + image + "/Ria/"
            for filename in os.listdir(R_dir):
                self.data_filenames.append(filename)

                R_filename = R_dir + filename
                Ria_filename = Ria_dir + filename
                R = sio.loadmat(R_filename)['R']
                Ria = sio.loadmat(Ria_filename)['Ria']
                np.save(self.tmp_dir_R + "filename", R)
                np.save(self.tmp_dir_Ria + "filename", Ria)
                
        # Train/Test split
        self.data_filenames_train, self.data_filenames_test = train_test_split(self.data_filenames, test_size=0.25, random_state=0)

        if trvaltest==0: # use train set
            self.data_filenames = self.data_filenames_train
        if trvaltest==1: # use val set
            self.data_filenames = self.data_filenames_test

    def __len__(self):
        # Number of samples
        return len(self.data_filenames)

    def __getitem__(self, idx):
        # Add flatten
        filename = self.data_filenames[idx]
        R = np.load(self.tmp_dir_R + filename, mmap_mode='r')
        Ria = np.load(self.tmp_dir_Ria + filename, mmap_mode='r')

        if self.transform:
            R = self.transform(R)
            Ria = self.transform(Ria)

        # trekant?
        sample = {'R': R,
                  'R_inv': Ria,
                  'filename': filename}
        
        return sample