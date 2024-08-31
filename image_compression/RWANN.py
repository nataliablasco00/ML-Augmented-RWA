from sklearn.linear_model import *
from sklearn.preprocessing import *
import math
import numpy as np
from joblib import dump, load
import time
from NeuralNetwork import *
from torch.cuda.amp import GradScaler, autocast

import torch

import os

import xgboost as xgb

def _entropy(data):
    """Compute the zero-order entropy of the provided data
    """
    values, count = np.unique(data.flatten(), return_counts=True)
    total_sum = sum(count)
    probabilities = (count / total_sum for value, count in zip(values, count))
    return -sum(p * math.log2(p) for p in probabilities)

def RWANN_Transform(raw_image, z, y, x, dtype, output, R=False, compression_technology='linear_regression', scale='all', train_split=0.01, verbose=False):
    G = np.fromfile(raw_image, sep="", dtype=dtype)
    im = np.reshape(G, (x*y, z), order="F").astype('int32')
    l = int(np.ceil(np.log2(z)))
    
    sifile = output[:-4] + '_SI.npy'

    RWAim = RWANN(im, l, sifile, R, compression_technology, scale, train_split, verbose)
    RWAim = RWAim.astype('int32')
    np.save(output, RWAim)

    print('\n Image: {} \n size: ({}, {}, {}) \n Transformed: {} \n'.format(raw_image, z, y, x, output));
    

def RWANN(im, l=1, sifile=None, R=False, compression_technology='linear_regression', scale=None, train_split=0.01,
          verbose=False, mode="individual_clustering", c=0, clusters=None, model_path=None, model_mode="RWA"):
    t1 = time.time()
    y, z = im.shape
    L, H = [], []
    fijo = None
    data = im.copy()


    for i in range(0, l):
        L, H = RWANN1l(data, sifile, R, compression_technology, scale, train_split, verbose, mode, c=c,
                       clusters=clusters, model_path=model_path, model_mode=model_mode)
        
        try:
            fijo = np.hstack((H, fijo))
        except ValueError:
            fijo = H.copy()
        data = L.copy()
        
    pim = np.hstack((L, fijo))
    
    return pim


def RWANN1l(im, sifile=None, R=False, compression_technology='linear_regression', scale=None, train_split=0.01,
            verbose=False, mode="individual_clustering", c=0, clusters=None, model_path=None, model_mode="RWA"):
    y, z = im.shape
    
    p = int(np.round(z / 2))
    q = int(np.floor(z / 2))
    
    if p % 2 == 0 and abs(p - (z/2)) > 0.4 and p == q:
        p += 1
    
    H = np.zeros((y, q))
    L = np.zeros((y, p))
    
    for j in range(0, q):
        H[:, j] = im[:, 2*j] - im[:, 2*j+1] # details
        L[:, j] = im[:, 2*j+1] + np.floor(H[:, j] / 2) # approximation
    
    if z % 2 != 0:
        L[:, -1] = im[:, -1]

    if L.shape[1] <= 100 or model_mode == "RWA":
        compression_technology = "linear_regression"
    elif (model_mode == "MLP" or model_mode == "S4") and L.shape[1] > 100:
        compression_technology = "NNRegressor"


    if compression_technology == 'linear_regression':
        M = fit_NNregression(L, H, sifile, R, compression_technology='linear_regression', train_split=train_split,
                             verbose=verbose, mode=mode, c=c, clusters=clusters, model_path=model_path,
                             model_mode=model_mode)
    else:
        M = fit_NNregression(L, H, sifile, R, compression_technology=compression_technology, train_split=train_split,
                             verbose=verbose, mode=mode, c=c, clusters=clusters, model_path=model_path,
                             model_mode=model_mode)
    
    H = H - np.round(M)
    H = H.astype("int32")

    def entropy_aux(data):
        data = torch.from_numpy(data)
        data = torch.round(data).int()

        unique_classes, counts = torch.unique(data, return_counts=True)
        total_sum = torch.sum(counts, dtype=torch.float)

        probabilities = counts / total_sum
        aux = probabilities * torch.log2(probabilities)
        entropy = -torch.sum(aux)

        return entropy


    print(" Level:", L.shape[1], "Entropy:", _entropy(H), "Entropy_2:", entropy_aux(H))
    
    return L, H


def fit_NNregression(X, y, sifile=None, R=False, compression_technology='linear_regression', train_split=0.01,
                     verbose=False, mode="individual_clustering", c=0, clusters=None, model_path=None, model_mode="RWA"):


    X_shape = X.shape[1]

    if R:
        X = np.reshape(X[:, 0], (X.shape[0], 1))

        poly = PolynomialFeatures(3)
        X = poly.fit_transform(X)


    if compression_technology == 'linear_regression':

        if not R:
            poly = PolynomialFeatures(1)
            X = poly.fit_transform(X)

        if mode == "individual_clustering":
            model = LinearRegression()
            model.fit(X, y)

            if not os.path.exists("output"):
                os.makedirs("output")

            dump(model, 'output//' + compression_technology + '_' + str(X_shape) + '.npy')

            model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        elif mode == "unique_clustering":
            model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')

        M = model.predict(X)


    elif compression_technology == "XGBoost":

        if mode == "individual_clustering":
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, learning_rate=0.5)
            model.fit(X, y)

            model.save_model("output//" + compression_technology + "_" + str(X_shape) + ".json")

        elif mode == "unique_clustering":
            model = xgb.XGBRegressor()
            model.load_model("output//" + compression_technology + "_" + str(X_shape) + ".json")

        M = model.predict(X)



    elif compression_technology == 'NNRegressor':

        model = torch.jit.load(model_path).to("cuda")
        model.eval()


        M = []
        batch_size = 5000
        with torch.no_grad():
            with autocast():
                min_val = 0
                max_val = 2 ** 16
                normalized_tensor = (X - min_val) / (max_val - min_val)

                for start_idx in range(0, normalized_tensor.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, normalized_tensor.shape[0])
                    if model_mode == "MLP":
                        output = model(torch.tensor(normalized_tensor[start_idx:end_idx, :]).float().to("cuda"))
                    elif model_mode == "S4":
                        output = model(torch.tensor(normalized_tensor[start_idx:end_idx, :]).float().to("cuda").unsqueeze(-1))
                    output = output.squeeze()
                    M.append(output.to("cpu"))

                M = torch.cat(M, dim=0).detach().numpy()


        M[np.isinf(M)] = 0
        M[np.isnan(M)] = 0
        M[M>65536] = 0

        poly = PolynomialFeatures(1)
        M = poly.fit_transform(M)

        model = LinearRegression()
        model.fit(M, y)
        dump(model, 'output//' + compression_technology + '_' + str(X_shape) + '.npy')

        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')

        M = model.predict(M)

    else:
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
    
        X = scaler.transform(X)
        
        M = model.predict(X)
        
    if verbose:
        print('Compressed scale ' + str(X_shape) + ' using ' + compression_technology)

    M = np.reshape(M, y.shape)

    return M



def inv_RWANN_Transform(raw_image, z, y, x, dtype, output, R, compression_technology='linear_regression', scale=-1):
    RWAim = np.load(raw_image, allow_pickle=True)
    l = int(np.ceil(np.log2(z)))
    
    sifile = raw_image[:-4] + '_SI.npy'
    
    im = inv_RWANN(RWAim, l, sifile, R, compression_technology, scale)
    np.save(output, im)
    
    print('\n Transformed: {} \n size: ({}, {}, {}) \n Recovered: {} \n'.format(raw_image, z, y, x, output));


def inv_RWANN(im, l=1, sifile=None, R=False, compression_technology='linear_regression', scale=-1, c=0,
              model_path=None, model_mode="RWA"):
    y, z = im.shape
    
    data = im
    
    P = []
    Q = []
    
    for i in range(0, l):
        p = int(np.ceil(z/2))
        q = int(np.floor(z/2))
        P.append(p)
        Q.append(q)
        
        z = p
    
    for i in reversed(range(0, l)):
        p = P[i]
        q = Q[i]
        
        L = data[:, :p]
        H = data[:, p:p+q]
                
        aux = inv_RWANN1l(L, H, sifile, R, compression_technology, scale, c=c, model_path=model_path, model_mode=model_mode)
        data[:, 0:p+q] = aux

    im=data
    
    return im
    
    
def inv_RWANN1l(L, H, sifile=None, R=False, compression_technology='linear_regression', scale=-1, c=0,
                model_path=None, model_mode="RWA"):

    if L.shape[1] <= 100 or model_mode == "RWA":
        compression_technology = "linear_regression"
    elif (model_mode == "MLP" or model_mode == "S4") and L.shape[1] > 100:
        compression_technology = "NNRegressor"

    if compression_technology == 'linear_regression':
        M = generate_NNregression(L, sifile, R, compression_technology='linear_regression', c=c, model_path=model_path,
                                  model_mode=model_mode)
    else:
        M = generate_NNregression(L, sifile, R, compression_technology=compression_technology, c=c, model_path=model_path,
                                  model_mode=model_mode)
    
    H = H + np.round(M)

    q = H.shape[1]
    p = L.shape[1]
    z = p+q

    im = np.zeros((L.shape[0], z))
    
    for j in range(0, q):
        im[:, 2*j+1] = L[:, j] - np.floor(H[:, j] / 2)
        
        im[:, 2*j] = im[:, 2*j+1] + H[:, j]
        
    if z % 2 != 0:
        im[:, 2*q] = L[:, -1]
        
    return im


def generate_NNregression(X, sifile=None, R=False, compression_technology='linear_regression', c=0, model_path=None,
                          model_mode="RWA"):

    X_shape = X.shape[1]

    if R:
        X = np.reshape(X[:, 0], (X.shape[0], 1))

        poly = PolynomialFeatures(3)
        X = poly.fit_transform(X)


    if compression_technology == 'linear_regression':

        if not R:
            poly = PolynomialFeatures(1)
            X = poly.fit_transform(X)
        
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')

        M = model.predict(X)

    elif compression_technology == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model("output//" + compression_technology + "_" + str(X_shape) + ".json")

        M = model.predict(X)

    elif compression_technology == 'NNRegressor':

        model = torch.jit.load(model_path).to("cuda")
        model.eval()

        batch_size = 5000
        M = []
        with torch.no_grad():
            with autocast():
                min_val = 0
                max_val = 2 ** 16
                normalized_tensor = (X - min_val) / (max_val - min_val)

                for start_idx in range(0, normalized_tensor.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, normalized_tensor.shape[0])
                    if model_mode == "MLP":
                        output = model(torch.tensor(normalized_tensor[start_idx:end_idx, :]).float().to("cuda"))
                    elif model_mode == "S4":
                        output = model(torch.tensor(normalized_tensor[start_idx:end_idx, :]).float().to("cuda").unsqueeze(-1))
                    output = output.squeeze()
                    M.append(output.to("cpu"))

                M = torch.cat(M, dim=0).detach().numpy()

        M[np.isinf(M)] = 0
        M[np.isnan(M)] = 0
        M[M > 65536] = 0

        poly = PolynomialFeatures(1)
        M = poly.fit_transform(M)

        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')

        M = model.predict(M)

    else:
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
        
        X = scaler.transform(X)
        M = model.predict(X)


    if len(M.shape) == 1:
        M = np.reshape(M, (M.shape[0], 1))
    
    return M    
            