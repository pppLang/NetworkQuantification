import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import sys


class dataset(Dataset):
    def __init__(self, mode='train'):
        if mode=='train':
            file_path = '/data0/langzhiqiang/cifar-10/data_batch_{}'
            self.data, self.labels = load_traindata(file_path=file_path)
        elif mode=='test':
            file_path = '/data0/langzhiqiang/cifar-10/test_batch'
            data_dict = unpickle(file_path)
            self.data = data_dict[b'data']
            self.labels = data_dict[b'labels']
        self.data = self.data/255
        self.num = len(self.labels)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index,:].reshape(3,32,32).astype(np.float32), self.labels[index]



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_traindata(file_path = '/data0/langzhiqiang/cifar-10/data_batch_{}'):
    train_data = None
    train_labels = None
    for i in range(5):
        data_dict = unpickle(file_path.format(i+1))
        if train_data is None:
            train_data = data_dict[b'data']
            train_labels = data_dict[b'labels']
        else:
            train_data = np.concatenate((train_data, data_dict[b'data']), axis=0)
            train_labels = np.concatenate((train_labels, data_dict[b'labels']), axis=0)
    # print(train_data.shape, train_labels.shape)
    return train_data, train_labels


def batch_info():
    file_path = 'C:\\Users\\user\\Documents\\ModelCompression\\Bi-Real-Net-by-me\\data_cifia10\\data_batch_1'
    data_dict = unpickle(file_path)
    print(data_dict.keys())

    data = np.array(data_dict[b'data'], dtype=np.uint8)
    labels = np.array(data_dict[b'labels'], dtype=np.uint8)
    print(data.shape)
    print(labels.shape)

    img1 = data[0,:].reshape(3,32,32)
    img1 = np.transpose(img1, [1,2,0])
    print(labels[0])
    plt.figure()
    plt.imshow(img1)
    plt.show()

    """ batch_label = data_dict[b'batch_label']
    filenames = data_dict[b'filenames']
    print(batch_label)
    print(len(filenames), filenames[0]) """


def test_batch_info():
    file_path = 'C:\\Users\\user\\Documents\\ModelCompression\\Bi-Real-Net-by-me\\data_cifia10\\test_batch'
    data_dict = unpickle(file_path)
    print(data_dict.keys())
    print(len(data_dict[b'labels']))


def dataset_info():
    file_path = 'C:\\Users\\user\\Documents\\ModelCompression\\Bi-Real-Net-by-me\\data_cifia10\\batches.meta'
    data_dict = unpickle(file_path)
    print(data_dict.keys())

    print(data_dict[b'num_cases_per_batch'])
    print(data_dict[b'label_names'])
    print(data_dict[b'num_vis'])


if __name__=='__main__':
    """ train_data, train_labels = load_traindata()
    print(sys.getsizeof(train_data[0,0]), train_data[0,0])
    print(sys.getsizeof(train_data[0,0:1]), train_data[0,0:1])
    # print(sys.getsizeof(train_data[0,0:105]), train_data[0,0:105])

    print(sys.getsizeof(train_data)) """
    test_batch_info()