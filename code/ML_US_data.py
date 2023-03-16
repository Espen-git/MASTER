import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

def scores(y_test, y_pred):
    """
    Prints MSE, R2 and MAD error scores for the predictions.

    input:
        - y_test
            True values.
        - y_pred
            predited values.
    output:
        - None
    """
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
    print(f"MAD: {mean_absolute_error(y_test, y_pred)}")

def accuracy(y_test, y_pred):
    """
    Rounds the prediction to neerest integer, then calculates accuracy.

    input:
        - y_test
            True values.
        - y_pred
            predited values.
    output:
        - None
    """
    y_pred = np.round(y_pred)
    res = y_pred==y_test
    print(np.sum(res) / len(y_test))

def run_models(X_train, X_test, y_train, y_test):
    """
    Trains and predicts using ML models.

    input:
        - X_train
            Traaining data
        - X_test
            Testing data
        - y_train 
            Training labels.
        - y_test
            Testing labels.
    output:
        - None
    """
    mlp = MLPRegressor(hidden_layer_sizes=(50,50), random_state=r, max_iter=100, tol=1*10^(-15)).fit(X_train, y_train)
    #hidden_layer_sizes
    #activation
    #solver
    #alpha=0.0001
    #batch_size
    #learning_rate_init=0.001
    
    mlp_y_pred = mlp.predict(X_test)
    print("MLP:")
    scores(y_test, mlp_y_pred)
    return mlp_y_pred


if __name__ == "__main__":
    data1 = sio.loadmat('R_Alpinion_L3-8_CPWC_hyperechoic_scatterers.mat') # load data from matlab
    data2 = sio.loadmat('R_Alpinion_L3-8_CPWC_hypoechoic.mat') # load data from matlab 
    data3 = sio.loadmat('R_Verasonics_P2-4_parasternal_long_small.mat') # load data from matlab 
    R1 = data1['R'] # shape: M x M x num_R
    R2 = data2['R']
    R3 = data3['R']
    R = np.concatenate((R1,R2,R3), axis=2).T # P x M x M
    P,M,M = R.shape
    
    X_real = R.real
    X_real = X_real.reshape((P, M*M))
    X_imag = R.imag
    X_imag = X_imag.reshape((P, M*M))
    X = np.concatenate((X_real,X_imag), axis=1) # Flattend R matrices, split in real and imaginary, shape P x M*M*2

    R_inv = np.linalg.inv(R) # P x M x M
    y_real = R_inv.real
    y_real = y_real.reshape((P, M*M))
    y_imag = R_inv.imag
    y_imag = y_imag.reshape((P, M*M))
    y = np.concatenate((y_real,y_imag), axis=1) # Flattend R inv matrices, split in real and imaginary, shape P x M*M*2
    print(X.shape)
    print(y.shape)
    
    r = 2022 # random state seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r)
    X_inv_pred = run_models(X_train, X_test, y_train, y_test)

    X_inv_pred_real, X_inv_pred_imag = np.split(X_inv_pred, 2, axis=1)
    X_inv_pred_real = X_inv_pred_real.reshape((X_inv_pred_real.shape[0], M, M))
    X_inv_pred_imag = X_inv_pred_imag.reshape((X_inv_pred_imag.shape[0], M, M))
    X_pred = np.empty((X_inv_pred_real.shape), dtype=np.cdouble) # Prediction of R inv for test set
    X_pred.real = X_inv_pred_real
    X_pred.imag = X_inv_pred_imag

    X_test_real, X_test_imag = np.split(X_test, 2, axis=1)
    X_test_real = X_test_real.reshape((X_test.shape[0], M, M))
    X_test_imag = X_test_imag.reshape((X_test.shape[0], M, M))
    X_test_reshape = np.empty((X_test_real.shape), dtype=np.cdouble) # R test set
    X_test_reshape.real = X_test_real
    X_test_reshape.imag = X_test_imag

    first_pred = X_pred[0,:,:]
    first_test = X_test_reshape[0,:,:]
    
    res1 = np.matmul(first_test, first_pred) # Should be identity
    plt.figure(1)
    plt.imshow(np.abs(res1))
    plt.colorbar()
    plt.title("First test X First pred")
    plt.show()

    second_test = X_test_reshape[1,:,:]
    res2 = np.matmul(second_test, first_pred) # Should not be identity
    plt.figure(2)
    plt.imshow(np.abs(res2))
    plt.colorbar()
    plt.title("Second test X First pred")
    plt.show()

    """
    third_test = X_test_reshape[2,:,:]
    res3 = np.matmul(third_test, first_pred) # Should not be identity
    plt.figure(3)
    plt.imshow(np.abs(res3))
    plt.colorbar()
    plt.title("Third test X First pred")
    plt.show()
    """

    # Testing random prediction agaist one R from each image source
    # Assumiong the one from the same image is most like identity??
    rand_int = np.random.randint(len(X_pred))
    random_pred = X_pred[rand_int,:,:]
    r1 = R1[:,:,0]
    r2 = R2[:,:,0]
    r3 = R3[:,:,0]

    res4 = np.matmul(r1, random_pred) 
    plt.figure(2)
    plt.imshow(np.abs(res4))
    plt.colorbar()
    plt.title("Second test X First pred")
    plt.show()

    res5 = np.matmul(r2, random_pred) 
    plt.figure(2)
    plt.imshow(np.abs(res5))
    plt.colorbar()
    plt.title("Second test X First pred")
    plt.show()

    res6 = np.matmul(r3, random_pred)
    plt.figure(2)
    plt.imshow(np.abs(res6))
    plt.colorbar()
    plt.title("Second test X First pred")
    plt.show()
