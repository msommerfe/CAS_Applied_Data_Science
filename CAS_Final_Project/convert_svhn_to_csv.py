import json
import os
import tensorflow as tf
#ds = tfds.load('huggingface:svhn/full_numbers')
from scipy.io import loadmat
import h5py
import numpy as np
import tensorflow_datasets as tfds
import h5py
f = h5py.File('/mnt/c/dev/test_digitStruct.mat', 'r')
list(f.keys())
print(f.keys())
dset = f['digitStruct']
size = f['/digitStruct/name'].size
print(f['/digitStruct/name'])
# Siehe auch: https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn




bboxs = f['digitStruct/bbox']


size = f['/digitStruct/name'].size

def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]]])

print(get_name(3, f))
print(get_box_data(10, f)['label'])