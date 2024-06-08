import json
import os
import tensorflow as tf
path = '/mnt/g/My Drive/development/datasets/OCR/MNIST_words_cropped/annotations.json'
#path = "//wsl.localhost/Ubuntu/mnt/g/My Drive/development/datasets/OCR/MNIST_words_cropped/annotations.json"
#path = 'G:/My Drive/development/datasets/OCR/MNIST_words_cropped/annotations.json'

print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(os.getcwd())

with open(path) as f:
    data = list(json.load(f).items())

print(data)