from TestModel import test_model
from itertools import product
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import joblib
import gc

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.utils import to_categorical


from RWAC import *


clusters = [1]
R = [False]
compression_technology = ["linear_regression"] # ["NNRegressor"] # ["XGBoost"]
train_split = [0.01]
mode = ["individual_clustering"] # ["unique_clustering"]
images_clustering = [1]

datasets_directory = "./../Data"
model_path = "./../trained_models/Best_S4_Hyperion.pth"
model_mode = "S4" # "MLP" # "RWA"

param_names = ["images_clustering", "n_clusters", "R", "compression_technology", "train_split", "mode"]

combinations = product(images_clustering, clusters, R, compression_technology, train_split, mode)

count_1 = 0
total = len(clusters) * len(R) * len(compression_technology) * len(train_split) * len(images_clustering)

for combination in combinations:
    count_1 += 1
    print("Starting iteration:", count_1, "/", total)
    params = dict(zip(param_names, combination))

    for root, _, files in os.walk(datasets_directory):

        count_2 = 0
        image = []
        for file in files:
            count_2 += 1

            if count_2 <= params["images_clustering"] and mode == ["unique_clustering"]:
                image_name = os.path.join(root, file).split('/')[-1]
                dtype = '>u2'
                z, x, y = image_name[:-4].split('x')[-3:]
                y = int(y.split('_')[0])
                x = int(x)
                z = int(z.split('-')[-1])

                G = np.fromfile(os.path.join(root, file), sep="", dtype=dtype)
                image_aux = np.reshape(G, (x * y, z), order="F").astype('int32')
                image.append(image_aux)
                del image_aux
                gc.collect()

                if count_2 == params["images_clustering"]:
                    image = np.vstack(image)

                    if params["n_clusters"] > 1:
                        km = KMeans(params["n_clusters"])
                        km_clusters = km.fit_predict(normalize(image))
                        joblib.dump(km, "kmeans.pkl")

                    output_folder = f'{"/".join(os.path.join(root, file).split("/")[:-3])}/output/{os.path.join(root, file).split("/")[-2]}_{params["n_clusters"]}_{R}_{compression_technology}_{mode}'
                    rwa_output = output_folder + '_RWA/'
                    os.makedirs(rwa_output, exist_ok=True)
                    rwa_output = rwa_output + image_name[:-4] + '.npy'

                    params_aux = params.copy()
                    params_aux["mode"] = "individual_clustering"

                    del params_aux["images_clustering"]
                    RWAC_Transform(os.path.join(root, file), z, y*params["images_clustering"], x, dtype, rwa_output, **params_aux, image=image)

                    del image
                    gc.collect()
            else:
                print("    Working on file", count_2, "/", len(files))
                full_path = os.path.join(root, file)

                test_model(full_path, scale=-1, verbose=False, write_results=True, model_path=model_path, model_mode=model_mode, **params)
                gc.collect()