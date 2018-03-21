import os
import gzip
import tarfile
import struct
import numpy as np


def get_data(label_file, data_file):
    with gzip.open(label_file, 'rb') as fin:
        struct.unpack(">II", fin.read(8))
        label = np.fromstring(fin.read(), dtype=np.uint8).astype(np.int32)
    with gzip.open(data_file, 'rb') as fin:
        struct.unpack(">IIII", fin.read(16))
        data = np.fromstring(fin.read(), dtype=np.uint8)
        data = data.reshape(len(label), 28, 28, 1)
    return data, label

if __name__ == '__main__':
    train_data  = get_data('C:\\Users\\mengtian\\.mxnet\\datasets\\fashion-mnist\\train-labels-idx1-ubyte.gz',
                           'C:\\Users\\mengtian\\.mxnet\\datasets\\fashion-mnist\\train-images-idx3-ubyte.gz')
    test_data = get_data('C:\\Users\\mengtian\\.mxnet\\datasets\\fashion-mnist\\t10k-labels-idx1-ubyte.gz',
                           'C:\\Users\\mengtian\\.mxnet\\datasets\\fashion-mnist\\t10k-images-idx3-ubyte.gz')
    print(train_data)