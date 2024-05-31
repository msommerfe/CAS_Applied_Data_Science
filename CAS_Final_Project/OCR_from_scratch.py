import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import cv2
import os
from sklearn.model_selection import train_test_split
import string
import csv
import util

from numpy import genfromtxt
from keras import backend as K
from keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
tf.config.run_functions_eagerly(True)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.sysconfig.get_build_info())

print(tf.test.is_built_with_cuda())

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




MAX_HIGHT, MAX_WIDTH, IMG_FOLDER, LABELS_File, ALPHABETS, MAX_STR_LEN, NUM_OF_CHARACTERS, NUM_OF_TIMESTAMPS, BATCH_SIZE = util.get_global_var()

keyVal = util.import_json_label_file()[:8000]
print(keyVal)

#make total path instead of just image name
key_val = util.make_total_path_for_all_image_names(keyVal)
print(key_val)

#Delete all values that are not in the alphabet
key_val = util.delete_key_values_that_not_in_alphabet(key_val)
print(key_val[0,0])


final_key_val = util.delete_key_values_with_too_small_aspect_ratio(key_val)

print(final_key_val.shape)