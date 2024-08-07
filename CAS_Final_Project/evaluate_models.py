import models
import util
import os
from os.path import isfile, join, basename
import glob


model = models.build_model()

#reading all key values from the folder on the drive
keyValMNIST = util.load_key_val_mnist()

# preprocessing all images that are in the key_val and convert them into the 4 Vectors required for ctc loss
x_all_img_total_path, labels_padded, len_labels_padded, len_labels_not_padded = (
    util.process_key_values_into_ctc_requierd_attributes(keyValMNIST))

train_dataset, validation_dataset = (
    util.create_tensorflow_train_and_validation_dataset(x_all_img_total_path, labels_padded, len_labels_padded, len_labels_not_padded, split_ratio = 0.0))




path_with_weights = '/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/Weights/simple_model/'
weight_files = glob.glob(join(path_with_weights, '*'))

for file in weight_files:
    model.load_weights(file)
    score = model.evaluate(validation_dataset, verbose=0)
    print(basename(file))
    print('Score: ' + str(score))

