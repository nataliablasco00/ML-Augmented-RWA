import numpy as np
from sklearn.cluster import KMeans

from RWA import *
from RWANN import *
from Entropy import entropy

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import joblib

def _entropy(data):
    """Compute the zero-order entropy of the provided data
    """
    values, count = np.unique(data.flatten(), return_counts=True)
    total_sum = sum(count)
    probabilities = (count / total_sum for value, count in zip(values, count))
    return -sum(p * math.log2(p) for p in probabilities)



from joblib import Parallel, delayed

"""
def compute_mi(x, y, i, j):
    print(i, j)
    return mutual_info_regression(x.reshape(-1, 1), y)[0]

# Calculate the mutual information matrix using parallel processing
def mutual_info_matrix(df):

    columns = df.columns
    mi_matrix = pd.DataFrame(index=columns, columns=columns)
    results = Parallel(n_jobs=-1)(delayed(compute_mi)(df.iloc[:, i].values, df.iloc[:, j].values, i, j)
                                  for i in range(len(columns)) for j in range(len(columns)) if i != j)
    k = 0
    for i in range(len(columns)):
        print(" -->", i)
        for j in range(len(columns)):
            print(" -->", j)
            if i == j:
                mi_matrix.iloc[i, j] = np.nan
            else:
                mi_matrix.iloc[i, j] = results[k]
                k += 1
    return mi_matrix.astype(float)"""


def compute_mi(i, j, df):
    print(i, j)
    x = df.iloc[:, i].values
    y = df.iloc[:, j].values
    mi = mutual_info_regression(x.reshape(-1, 1), y).item()
    return i, j, mi


# Compute the upper triangular part of the mutual information matrix in parallel
def mutual_info_matrix(df):
    columns = df.columns
    n = len(columns)
    mi_matrix = pd.DataFrame(index=columns, columns=columns)

    # List of index pairs for the upper triangular matrix
    index_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Compute mutual information in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_mi)(i, j, df) for i, j in index_pairs)

    # Fill the mutual information matrix with results
    for i, j, mi in results:
        mi_matrix.iloc[i, j] = mi
        mi_matrix.iloc[j, i] = mi  # Ensure symmetry

    for i in range(n):
        mi_matrix.iloc[i, i] = 0

    print(mi_matrix.dtypes)
    print(mi_matrix.isna().sum())
    mi_matrix = mi_matrix.fillna(0)

    return mi_matrix

def RWAC_Transform(raw_image, z, y, x, dtype, output, n_clusters=4, R=False, compression_technology='linear_regression',
                   layer=-1, train_split=0.01, verbose=False, mode="individual_clustering", image=None,
                   model_path=None, model_mode="RWA"):

    if image is None:
        G = np.fromfile(raw_image, dtype=dtype, count=x*y*z)
        image = np.reshape(G, (x*y, z), order="F") # .astype('int32')

    new_image = np.empty_like(image)
    l = int(np.ceil(np.log2(z)))


    if n_clusters > 1:
        if mode == "individual_clustering":
            km = KMeans(n_clusters)
            #km_clusters = km.fit_predict(normalize(image[:, 1:] - image[:, :-1]))
            image_aux = image.unsqueeze(-1)
            image_aux = image_aux.reshape(-1, 2)
            print(image.shape, image_aux.shape)
            km_clusters = km.fit_predict(normalize(image_aux[:, :, 1] - image_aux[:, :, 0], norm="l2"))


        elif mode == "unique_clustering":
            km = joblib.load("kmeans.pkl")
            km_clusters = km.predict(normalize(image))
            # km_clusters = load_and_predict(image)
            print("        --> number of clusters used:", len(np.unique(km_clusters)), "/", n_clusters)
        else:
            print("Error: mode not defined")
            return None
        km_clusters_aux = km_clusters

    else:
        km_clusters = np.zeros_like(image[:, 0])
        km_clusters_aux = km_clusters


    np.save(output[:-4] + '_KM.npy', km_clusters)
    _final_output = output[:-4] + '_CL_final.raw'
    
    # compression
    for c in range(n_clusters):
        if c not in np.unique(km_clusters):
            continue
        if verbose:
            print('Cluster ' + str(c))
        
        _sifile = output[:-4] + '_CL_' + str(c) + '_SI.npy'
        _output = output[:-4] + '_CL_' + str(c) + '.npy'
        _output_raw = output[:-4] + '_CL_' + str(c) + '.raw'
        
        im = image[km_clusters == c].astype('int32')

        RWAim = RWANN(im, l, _sifile, R, compression_technology, layer, train_split, verbose, mode=mode, c=c,
                      clusters=km_clusters_aux, model_path=model_path, model_mode=model_mode)
        RWAim = RWAim.astype('int32')

        np.save(_output, RWAim)

        new_image[km_clusters == c] = RWAim

    np.save(output, new_image)



    RWAim = np.reshape(RWAim, (x, y, z), order="F")
    np.asfortranarray(RWAim).T.astype(dtype).tofile(_final_output)

    return _entropy(new_image)
    

def inv_RWAC_Transform(raw_image, z, y, x, dtype, output, R=False, compression_technology='linear_regression', layer=-1,
                       std_param=None, model_path=None, model_mode="RWA"):
    km_clusters = np.load(raw_image[:-4] + '_KM.npy')
    n_clusters = np.unique(km_clusters)
    recovered_image = np.empty(shape=(km_clusters.shape[0], z), dtype='int32')
    l = int(np.ceil(np.log2(z)))

    output_raw = output[:-4] + ".raw"
    for c in n_clusters:
        _raw_image = raw_image[:-4] + '_CL_' + str(c) + '.npy'
        _sifile = raw_image[:-4] + '_CL_' + str(c) + '_SI.npy'
        _output = output[:-4] + '_CL_' + str(c) + '.npy'

        try:
            RWAim = np.load(_raw_image)
        except:
            continue

        im = inv_RWANN(RWAim, l, _sifile, R, compression_technology, layer, c, model_path=model_path, model_mode=model_mode)
        np.save(_output, im)

        recovered_image[km_clusters == c] = im

    np.save(output, recovered_image)

    recovered_image = np.reshape(recovered_image, (x*y, z), order="F")
    recovered_image = np.asfortranarray(recovered_image)
    recovered_image.astype(dtype).tofile(output_raw)
