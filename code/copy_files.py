import scipy.io as sio
import os
        
root_dir = 'C:/Users/espen/Documents/Skole/MASTER/code/'
data_dir = root_dir + "data/"
verasonics = 'Verasonics_P2-4_parasternal_long_small'
verasonics_1 = 'Verasonics_P2-4_parasternal_long_small_1_frame'
verasonics_4 = 'Verasonics_P2-4_parasternal_long_small_4_frames'


R_dir = data_dir + verasonics + "/R/"
R_dir_1 = data_dir + verasonics_1 + "/R/"
R_dir_4 = data_dir + verasonics_4 + "/R/"

Ria_dir = data_dir + verasonics + "/Ria/"
Ria_dir_1 = data_dir + verasonics_1 + "/Ria/"
Ria_dir_4 = data_dir + verasonics_4 + "/Ria/"

for filename in os.listdir(R_dir):
    filepath = R_dir + filename
    R = sio.loadmat(filepath)
    
    if filename[0] == '1':
        save_path = R_dir_1 + '/' + filename
        sio.savemat(save_path, R)
    else:
        save_path = R_dir_4 + '/' + filename
        sio.savemat(save_path, R)

print("R done")

for filename in os.listdir(Ria_dir):
    filepath = Ria_dir + filename
    Ria = sio.loadmat(filepath)
    
    if filename[0] == '1':
        save_path = Ria_dir_1 + '/' + filename
        sio.savemat(save_path, Ria)
    else:
        save_path = Ria_dir_4 + '/' + filename
        sio.savemat(save_path, Ria)