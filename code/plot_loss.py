import matplotlib.pyplot as plt
import numpy as np

def plot(model):
    data_path = 'C:/Users/espen/Documents/Skole/MASTER/code/data/models/' + model + '.npz'
    data = np.load(data_path) 
    val_losses = data['vallosses']; train_losses = data['trainlosses']

    plt.plot(train_losses, label='Traning', color='red', linestyle='dashed')
    plt.plot(val_losses, label='Validation', color='blue', linestyle='solid')
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()

if __name__=='__main__':
    plot('Test5(Two images)')