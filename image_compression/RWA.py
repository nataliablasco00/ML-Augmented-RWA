import os
import numpy as np


def RWA_Transform(raw_image, z, y, x, dtype, output):
    G = np.fromfile(raw_image, sep="", dtype=dtype)
    im = np.reshape(G, (x*y, z), order="F")
    l = int(np.ceil(np.log2(z)))

    RWAim, W = RWA(im, l, 1)
    RWAim = RWAim.astype('int32')
    np.save(output, RWAim)
    
    sifile = output[:-4] + '_SI.npy'
    np.save(sifile, W)

    print('\n Image: {} \n size: ({}, {}, {}) \n Transformed: {} \n side information: {} \n'.format(raw_image, z, y, x, output, sifile));
    

def RWA(im, l, n):
    y, z = im.shape
    L, H, W, MSE = [], [], [], []
    fijo = None
    data = im.copy()

    for i in range(0, l):
        L, H, w, mse = RWA1l(data, n)
        
        W.append(w) # W[i] = w
        MSE.append(mse) # MSE[i] = mse
        
        try:
            fijo = np.hstack((H, fijo))
        except ValueError:
            fijo = H.copy()
        data = L.copy()
    
    pim = np.hstack((L, fijo))
    
    return pim, W


def RWA1l(im, n):        
    y, z = im.shape
    
    p = int(np.round(z / 2))
    q = int(np.floor(z / 2))
    
    if p % 2 == 0 and abs(p - (z/2)) > 0.4 and p == q:
        p += 1
    
    H = np.zeros((y, q))
    L = np.zeros((y, p))
    
    for j in range(0, q):
        H[:, j] = im[:, 2*j] - im[:, 2*j+1] # details
        L[:, j] = im[:, 2*j+1] + np.floor(H[:, j] / 2) # residuals
    
    if z % 2 != 0:
        L[:, -1] = im[:, -1]
        
    M, W = fit_regression(L, H, n)
    
    mse = np.sum((M - H)**2) / H.shape[0] / H.shape[1]
    H = H - np.round(M)
    
    return L, H, W, mse


def fit_regression(x, y, order):
    V = np.ones((x.shape[0], 1))

    for j in range(1, order+1):
        V = np.hstack((V, x**j))

    W = np.matmul(np.linalg.pinv(V), y)

    M = np.matmul(V, W) # TODO: improve

    return M, W


def inv_RWA_Transform(raw_image, z, y, x, dtype, output, sifile):
    RWAim = np.load(raw_image, allow_pickle=True)
    
    
    l = int(np.ceil(np.log2(z)))
    W = np.load(sifile, allow_pickle=True)
    
    im = inv_RWA(RWAim, l, W, 1)
        
    im.transpose().astype(dtype).tofile(output)
    
    print('\n Transformed: {} \n size: ({}, {}, {}) \n Recovered: {} \n side information: {} \n'.format(raw_image, z, y, x, output, sifile));


def inv_RWA(im, l, W, n):
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
        
        w = W[i]
        
        aux = inv_RWA1l(L, H, w, n)
        data[:, 0:p+q] = aux

    im=data
    
    return im
    
    
def inv_RWA1l(L, H, W, n):
    M = generate_regression(L, W, n)
    
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


def generate_regression(x, W, order):
    V = np.ones((x.shape[0], 1))

    for j in range(1, order+1):
        V = np.hstack((V, x**j))

    M = np.matmul(V, W) # TODO: improve

    return M

# raw_image, z, y, x, dtype, output = 'Image_30x1528x60_u16.raw', 30, 1528, 60, '>i2', 'output/RWA.raw'
# RWA_Transform(raw_image, z, y, x, dtype, output)
# raw_image, z, y, x, dtype, output, sifile = 'output/RWA.raw', 30, 1528, 60, '>i2', 'output/inv_RWA.raw', 'output/RWA_SI.npy'
# inv_RWA_Transform(raw_image, z, y, x, dtype, output, sifile)
