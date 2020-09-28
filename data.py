import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from scipy import signal as sig
from pathlib import Path
from net import RNN



# LOADING
def check_load(p="project_datasets/"):
    return Path(p+"pre_data.mat").exists()

def save_pre_processed(X, y, Xval, yval, Xtest, ytest, p="project_datasets/"):
    dataset = h5py.File(p+'pre_data.mat', 'w')
    dataset.create_dataset('train_x', data=X)
    dataset.create_dataset('train_y', data=y)

    dataset.create_dataset('val_x', data=Xval)
    dataset.create_dataset('val_y', data=yval)

    dataset.create_dataset('test_x', data=Xtest)
    dataset.create_dataset('test_y', data=ytest)

    dataset.close()



def load_pre_processed(folder_path="project_datasets/"):
    dataset = h5py.File(folder_path+'pre_data.mat', 'r')

    X = np.copy(dataset.get('train_x'))
    y = np.copy(dataset.get('train_y'))

    Xv = np.copy(dataset.get('val_x'))
    yv = np.copy(dataset.get('val_y'))

    Xt = np.copy(dataset.get('test_x'))
    yt = np.copy(dataset.get('test_y'))

    return X, y, Xv, yv, Xt, yt




def load(number, folder_path="project_datasets/"):
    folder_path += 'A0%dT_slice.mat' % number
    A01T = h5py.File(folder_path, 'r')
    X = np.copy(A01T['image'])
    y = np.copy(A01T['type'])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    return X, y


def recode_y(y):
    return np.subtract(y, np.min(y))


# PREPROCESS

def add_noise(X, std=0.02):
    G = np.random.normal(0, std, X.shape)
    return np.add(G, X)


def normalize_data(X, mean=None, var=None, axis=0, kdims=True):
    if mean is None or var is None:
        mean = np.mean(X, axis=axis, keepdims=kdims)
        var = np.var(X, axis=axis, keepdims=kdims)


    return np.divide(np.subtract(X, mean), np.sqrt(var) + 1e-7), mean, var


def standardize_data(X, mean=None, var=None, axis=None):
    if mean is None or var is None:
        mean = np.mean(X)
        var = np.var(X)

    return np.divide(np.subtract(X, mean), np.sqrt(var) + 1e-7), mean, var

def percentile(X, p=5):
    q1 = np.percentile(X, p)
    q3 = np.percentile(X, 100-p)

    X[X < q1] = q1
    X[X > q3] = q3

    return X


def butter_filter(X, hz=4, filter='highpass', order=3):
    f = sig.butter(order, hz / 125, filter, False, 'ba')

    return sig.lfilter(f[0], f[1], X, axis=2)


def butter_band(X, hzl, hzh, filter='bandpass', order=3):
    f = sig.butter(order, (hzl/125, hzh/125), filter, False, 'ba')
    return sig.lfilter(f[0], f[1], X, axis=2)


def drop_nan(X, y):
    idx = np.unique(np.argwhere(np.isnan(X))[:,0])

    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    return  X, y


def expand_dims(X, Xv, Xt):
    return np.expand_dims(X, 3), np.expand_dims(Xv, 3), np.expand_dims(Xt, 3)


def skip_in(X, n=125):
    return X[:, :, n:]


def swap_axis(X):
    return np.moveaxis(X, 1, 2)

# AUGMENT

def augment_noise(X, y, p=0.25):
    N = X.shape[0]
    added = int(N * p)
    idx = np.arange(0, N)
    np.random.shuffle(idx)

    Xtra = X[idx[:added]]
    noise = np.random.normal(1, 0.05, (added, X.shape[1], X.shape[2]))
    Xtra = np.multiply(Xtra, noise)

    Xtra = np.concatenate((X, Xtra), axis=0)
    ytra = np.concatenate((y, y[idx[:added]]), axis=0)

    return Xtra, ytra


def windowing(X, y, n_start=0, window=200, stride=50):
    N, C, T = X.shape

    s0 = int((T-window - n_start) / stride) + 1

    Xf = np.zeros((s0*N, C, window))
    t = 0
    while(n_start + t*stride + window <= T):
        Xf[t*N:(t+1)*N] = X[:, :, n_start+t*stride:n_start+t*stride+window]
        t += 1

    return Xf, np.vstack([y]*t)



def augment_frequency(X, freqz, ceiling=True):
    N = len(freqz)
    a, b, c = X.shape
    if ceiling:
        d = b * N
    else:
        d = b * (N-1)

    Xf = np.zeros((a, d, c))
    for i in range(0, N):
        if i == N-1:
            if ceiling:
                Xf[:, i * b:(i + 1) * b, :] = butter_band(X, freqz[i], 124)
        else:
            Xf[:, i * b:(i + 1) * b, :] = butter_band(X, freqz[i], freqz[i + 1])

    return Xf


def diff(X, o = 1,axis=2):
    a, b, c = X.shape
    Xf = np.zeros((a, (1+o)*b, c))
    Xf[:, :b, :] = X
    Xf[:, b:2*b, :] = np.gradient(X,axis=axis)
    if o ==2:
        Xf[:, 2*b:, :] = np.gradient(Xf[:, b:2*b, :], axis=axis)

    return Xf


def power(X, w=40, axis=2, l=22):
    a, b, c = X.shape
    Xf = np.zeros((a, b+l, c))
    p = np.array([np.sum(np.square(X[:, :l, i-(w-1):i+1]), axis=axis) if i>(w-1) else np.sum(np.square(X[:, :l, :i+1]), axis=axis)  for i in range(c)])
    Xf[:, b:, :] = np.swapaxes(np.swapaxes(p/w**2, 0, 2), 0, 1)

    return Xf


def shuffle(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]