
import torch
import copy
import math
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

from sklearn.linear_model import *
from sklearn.preprocessing import *

import gc



def train_single(model, criterion, optimizer_list, epochs, train_loader, val_loader, entropy_criterion, l, device,
                 scaler_list, level, scheduler, mode):
    best_entropy = 10000000
    best_entropy_val = 10000000
    epochs_best = 0

    for epoch in range(epochs):
        epochs_best += 1
        sum_entropy = 0


        model.train()

        for batch_data in train_loader:
            loss = 0
            signal = batch_data
            idx = torch.randperm(signal.shape[0])
            signal = signal[idx, :]

            del idx
            z = batch_data.shape[-1]

            for i in range(level):
                optimizer = optimizer_list[-1]
                model.train()

                y, z = signal.shape
                p = int(np.round(z / 2))
                q = int(np.floor(z / 2))

                if p % 2 == 0 and abs(p - (z / 2)) > 0.4 and p == q:
                    p += 1

                H = np.zeros((y, q))
                L = np.zeros((y, p))

                for j in range(0, q):
                    H[:, j] = signal[:, 2 * j] - signal[:, 2 * j + 1]  # details
                    L[:, j] = signal[:, 2 * j + 1] + np.floor(H[:, j] / 2)  # approximation

                if z % 2 != 0:
                    L[:, -1] = signal[:, -1]



                batch_size = int(813056/750)

                if i == level-1:

                    del signal, batch_data
                    gc.collect()

                    residuals_level = []
                    idx = 0

                    for start_idx in range(0, L.shape[0], batch_size):
                        end_idx = min(start_idx + batch_size, L.shape[0])
                        new_H = H[start_idx:end_idx, :]
                        new_L = L[start_idx:end_idx, :]
                        idx += 1

                        #print(idx)

                        with autocast():
                        #if True:
                            min_val = 0
                            max_val = 2 ** 16

                            normalized_tensor = torch.tensor((new_L - min_val) / (max_val - min_val)).float()

                            if mode == "MLP":
                                outputs = model(normalized_tensor.to(device))
                            elif mode == "S4":
                                outputs = model(normalized_tensor.unsqueeze(-1).to(device))

                            outputs = outputs[:, :new_H.shape[1]]
                            loss = criterion(outputs.squeeze().float(), torch.tensor(new_H).float().to(device))

                        optimizer.zero_grad()
                        scaler_list[-1].scale(loss.float()).backward()
                        scaler_list[-1].step(optimizer)
                        scaler_list[-1].update()

                        optimizer_list[-1] = optimizer

                        if len(outputs.shape) == 3:
                            outputs = outputs.squeeze(-1)
                        res = torch.tensor(new_H).to(device) - torch.round(outputs)
                        residuals_level.append(res.to("cpu"))

                    residuals_level = torch.cat(residuals_level, dim=0)
                else:
                    signal = L

            sum_entropy += entropy_criterion(residuals_level.to("cpu")).item()

        del residuals_level, res, outputs, new_H, new_L, H, L
        gc.collect()

        model.eval()
        sum_entropy_val = 0
        for batch_data in val_loader:
            signal = batch_data[:500000, :]
            if sum_entropy_val > 0:
                break
            z = batch_data.shape[-1]

            for i in range(level):

                y, z = signal.shape
                p = int(np.round(z / 2))
                q = int(np.floor(z / 2))

                if p % 2 == 0 and abs(p - (z / 2)) > 0.4 and p == q:
                    p += 1

                H = np.zeros((y, q))
                L = np.zeros((y, p))

                for j in range(0, q):
                    H[:, j] = signal[:, 2 * j] - signal[:, 2 * j + 1]  # details
                    L[:, j] = signal[:, 2 * j + 1] + np.floor(H[:, j] / 2)  # approximation

                if z % 2 != 0:
                    L[:, -1] = signal[:, -1]

                if i == level-1:

                    residuals_level = []
                    idx = 0


                    for start_idx in range(0, L.shape[0], batch_size):
                        end_idx = min(start_idx + batch_size, L.shape[0])
                        new_H = H[start_idx:end_idx, :]
                        new_L = L[start_idx:end_idx, :]
                        idx += 1

                        with torch.no_grad():
                            with autocast():
                            #if True:
                                min_val = 0
                                max_val = 2 ** 16
                                normalized_tensor = torch.tensor((new_L - min_val) / (max_val - min_val)).float()

                                if mode == "MLP":
                                    outputs = model(normalized_tensor.to(device))
                                if mode == "S4":
                                    outputs = model(normalized_tensor.to(device).unsqueeze(-1))

                                outputs = outputs[:, :new_H.shape[1]]


                        if len(outputs.shape) == 3:
                            outputs = outputs.squeeze(-1)
                        res = torch.tensor(new_H).to(device) - torch.round(outputs)
                        residuals_level.append(res.to("cpu"))

                    residuals_level = torch.cat(residuals_level, dim=0)

                else:
                    signal = L

            del signal, outputs, res, H, L, new_H, new_L
            sum_entropy_val += entropy_criterion(residuals_level.to("cpu")).item()
            del residuals_level
            gc.collect()

        print(f'Epoch [{epoch + 1}/{epochs}], Entropy Train: {sum_entropy / len(train_loader)} \t Entropy Val: {sum_entropy_val / len(val_loader)} \t Loss: {loss}')
        scheduler.step(sum_entropy / len(train_loader))

        if sum_entropy_val / len(val_loader) < best_entropy_val and math.isnan(loss) == False:
            best_entropy_val = sum_entropy_val / len(val_loader)
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        if sum_entropy / len(train_loader) < best_entropy and math.isnan(loss) == False:
            best_entropy = sum_entropy / len(train_loader)
            epochs_best = 0



        if epochs_best >= 20 or math.isnan(loss) or math.isinf(loss):
            break


    return best_model, best_entropy_val,  best_epoch

