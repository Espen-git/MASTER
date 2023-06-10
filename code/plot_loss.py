import matplotlib.pyplot as plt
import numpy as np
import scienceplots

def plot(model):
    data_path = 'C:/Users/espen/Documents/Skole/MASTER/code/data/models/' + model + '.npz'
    data = np.load(data_path) 
    val_losses = data['vallosses']; train_losses = data['trainlosses']

    with plt.style.context(['science','grid']):
        fig, ax = plt.subplots()
        ax.plot(train_losses, label='Traning', color='red', linestyle='dashed')
        ax.plot(val_losses, label='Validation', color='blue', linestyle='solid')
        ax.autoscale(tight=True)
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.legend()
        #plt.show()
        fig.savefig('images/lossplot_' + model + '.svg', dpi=300)
        plt.close()

def print_config(model):
    data_path = 'C:/Users/espen/Documents/Skole/MASTER/code/data/models/' + model + '.npz'
    data = np.load(data_path, allow_pickle=True)

    print(data['config'])


if __name__=='__main__':
    plot("Test2")
    plot('Test3(upper_triangular)')
    plot('Test4')
    plot('Test5(Two_images)')
    plot('Test6(hypoechoic)')
    plot('Test7(hyperechoic)')
    plot('Test8(Verasonics)')
    plot('Test9(Verasonics-complex_network)')
    plot('Test10(Verasonics-complex_network2)')
    plot('Test11(Verasonics-complex_network_4_frames)')
    plot('Test12(Verasonics-small_network_4_frames)')
    #print_config("Test3(upper_triangular)")