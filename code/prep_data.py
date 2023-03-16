# Split data
import numpy as np
import scipy.io as sio
import os.path
import matplotlib.pyplot as plt

Alpinion_hyperechoic_scatterers_data = sio.loadmat('R_Alpinion_L3-8_CPWC_hyperechoic_scatterers.mat') # load data from matlab
Alpinion_hypoechoic_data = sio.loadmat('R_Alpinion_L3-8_CPWC_hypoechoic.mat') # load data from matlab 
Verasonics_parasternal_long_small_data = sio.loadmat('R_Verasonics_P2-4_parasternal_long_small.mat') # load data from matlab 

R1 = Alpinion_hyperechoic_scatterers_data['R'] # shape: M x M x num_R
R2 = Alpinion_hypoechoic_data['R']
R3 = Verasonics_parasternal_long_small_data['R']

# scale data

data_dir = "C:/Users/espen/Documents/Skole/MASTER/code/data/"
data_string1 = "Alpinion_hyperechoic_scatterers_data"
data_string2 = "Alpinion_hypoechoic_data"
data_string3 = "Verasonics_parasternal_long_small_data"

length_R1 = R1.shape[2]
length_R2 = R2.shape[2]
length_R3 = R3.shape[2]

def data1():
    for i in range(length_R1):
        R = R1[:,:,i]
        R_inv = np.linalg.inv(R)
        filename = data_string1 + "_" + str(i) + ".npy"
        completeName = os.path.join(data_dir, filename)
        both_R = np.stack((R, R_inv), axis=2)
        np.save(completeName, both_R)

def data2():
    for i in range(length_R2):
        R = R2[:,:,i]
        R_inv = np.linalg.inv(R)
        filename = data_string2 + "_" + str(i) + ".npy"
        completeName = os.path.join(data_dir, filename)
        both_R = np.stack((R, R_inv), axis=2)
        np.save(completeName, both_R)

def data3():
    for i in range(length_R3):
        R = R3[:,:,i]
        R_inv = np.linalg.inv(R)
        filename = data_string3 + "_" + str(i) + ".npy"
        completeName = os.path.join(data_dir, filename)
        both_R = np.stack((R, R_inv), axis=2)
        np.save(completeName, both_R)


# Delete old files before running again!!
data1()
#data2()
#data3()

#print(both_R[:,:,0]) # R
#print(both_R[:,:,1]) # R_inv