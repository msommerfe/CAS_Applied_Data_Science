import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import tensorflow as tf
#from PIL import Image
#import cv2
import os
from sklearn.model_selection import train_test_split
import string
import csv
import util

from numpy import genfromtxt
from keras import backend as K
from keras.models import Model
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
tf.config.run_functions_eagerly(True)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




MAX_HIGHT, MAX_WIDTH, IMG_FOLDER, LABELS_File, ALPHABETS, MAX_STR_LEN, NUM_OF_CHARACTERS, NUM_OF_TIMESTAMPS, BATCH_SIZE = util.get_global_var()

IMG_FOLDER = '/mnt/g/My Drive/development/datasets/OCR/MNIST_words_cropped/images/'


keyValMNIST = util.import_json_label_file(path='/mnt/g/My Drive/development/datasets/OCR/MNIST_words_cropped/annotations.json')
keyValMNIST = util.make_total_path_for_all_image_names(keyValMNIST, path= '/mnt/g/My Drive/development/datasets/OCR/MNIST_words_cropped/images/')
print(keyValMNIST.shape)
print(keyValMNIST)


keyValch4 = util.import_txt_csv_label_file(path='/mnt/g/My Drive/development/datasets/OCR/ch4_cropped/annotations.txt')
keyValch4 = util.make_total_path_for_all_image_names(keyValch4, path= '/mnt/g/My Drive/development/datasets/OCR/ch4_cropped/images/')
print(keyValch4.shape)
print(keyValch4)

keyValBd = util.import_txt_csv_label_file(path='/mnt/g/My Drive/development/datasets/OCR/BornDigitalData/annotations.txt')
keyValBd = util.make_total_path_for_all_image_names(keyValBd, path= '/mnt/g/My Drive/development/datasets/OCR/BornDigitalData/images/')
print(keyValBd.shape)
print(keyValBd)

keyVal100k = util.import_txt_csv_label_file(path = "/mnt/g/My Drive/development/datasets/OCR/tr_synth_100K_cropped/annotations.txt")
keyVal100k = util.make_total_path_for_all_image_names(keyVal100k, path= '/mnt/g/My Drive/development/datasets/OCR/tr_synth_100K_cropped/images/')
print(keyVal100k.shape)
print(keyVal100k)

keyValIdVehic = util.import_txt_csv_label_file(path = "/mnt/g/My Drive/development/datasets/OCR/Ind_vehicle_number/annotations.txt")
keyValIdVehic = util.make_total_path_for_all_image_names(keyValIdVehic, path= '/mnt/g/My Drive/development/datasets/OCR/Ind_vehicle_number/images/')
print(keyValIdVehic.shape)
print(keyValIdVehic)


key_val = np.concatenate((keyValMNIST,  keyValch4, keyValBd,  keyVal100k, keyValIdVehic ), axis=0)

#key_val = keyValIdVehic
#shuffles the keyVals
np.random.shuffle(key_val)
print(key_val.shape)

#Delete all values that are not in the alphabet
key_val = util.delete_key_values_that_not_in_alphabet(key_val)
print(key_val[0,0])
print(key_val.shape)

final_key_val= util.delete_key_values_that_have_a_too_long_label(key_val)


#final_key_val = util.delete_key_values_with_too_small_aspect_ratio(key_val)
print("Totals, Number of used images for training and validation"+   str(final_key_val.shape))


 #Convert key Value (x = imagePpath y = label) to np array
x_all_img_total_path = final_key_val[:,0]
labels = final_key_val[:,1]

# add the total path to each image name
#x_all_img_total_path = np.array([make_total_path(imgName) for imgName in x_all_img_file_name])

#Convert String to int-Code including padding the int code to max str len
labels_padded = np.array([util.label_to_num(xi) for xi in labels])
len_labels_padded = np.array([len(i) for i in labels_padded])
len_labels_not_padded = np.array([len(i) for i in labels])

print(x_all_img_total_path[10])
print(x_all_img_total_path.shape)
print(labels_padded.shape)
print(len_labels_padded.shape)
print(len_labels_not_padded.shape)


characters = set()
for item in labels:
    for ch in item:
        characters.add(ch)

# Sort the characters
characters = sorted(characters)
print("Characters present: ", str(characters))
print("Max anzahl Character in labels:", str(len_labels_not_padded.max()))



dataset = tf.data.Dataset.from_tensor_slices((x_all_img_total_path, labels_padded, len_labels_padded, len_labels_not_padded))

# Split dataset into training and validation dataset
dataset_size = len(x_all_img_total_path)
train_size = int(0.9 * dataset_size)

train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

#Apply the process_single_sample function to each item in train and validation dataset
#Bewusst erst Dataset erstellt und dann einmal map für train und validation. Da ich nicht weiss wie sich das .Batch auswirkt wenn ich es nur einmal auf den ganzen Datensatz anwende und anschliessend splitte
train_dataset = (
    train_dataset.map(
        util.process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = (
    validation_dataset.map(
        util.process_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)




_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    len_labels_padded = batch["input_length"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = util.num_to_label(labels[i])
        print(len_labels_padded[i])
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()

figures_list = []


class PlotPredictions(tf.keras.callbacks.Callback):

    def __init__(self, frequency=1):
        self.frequency = frequency
        super(PlotPredictions, self).__init__()

        batch = validation_dataset.take(1)
        self.batch_images = list(batch.as_numpy_iterator())[0]["image"]
        self.batch_labels = list(batch.as_numpy_iterator())[0]["label"]

    def plot_predictions(self, epoch):

        prediction_model = keras.models.Model(
            self.model.input[0],
            self.model.get_layer(name="dense2").output
        )

        preds = prediction_model.predict(self.batch_images)
        preds = preds[:, :-2]
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        pred_texts = keras.backend.ctc_decode(preds,
                                        input_length=input_len,
                                        greedy=True)[0][0]


        fig, ax = plt.subplots(4, 4, figsize=(15, 5))
        fig.suptitle('Epoch: ' + str(epoch), weight='bold', size=14)

        for i in range(16):
            img = (self.batch_images[i, :, :, 0] * 255).astype(np.uint8)
            title = f"Prediction: {util.num_to_label(pred_texts[i])}"
            ax[i // 4, i % 4].imshow(img.T, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

        plt.show()
        # plt.savefig("predictions_epoch_"+ str(epoch)+'.png', bbox_inches = 'tight', pad_inches = 0)

        figures_list.append(fig)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            self.plot_predictions(epoch)











class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # On test time, just return the computed loss
        return loss



def build_model():
    # Inputs to the model
    input_img = layers.Input(shape=(MAX_WIDTH, MAX_HIGHT, 1),
                            name="image",
                            dtype='float32')
    labels = layers.Input(name="label", shape=[MAX_STR_LEN], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # First conv block
    x = layers.Conv2D(32,
               (3,3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same',
               name='Conv1')(input_img)
    x = layers.MaxPooling2D((2,2), name='pool1')(x)

    # Second conv block
    x = layers.Conv2D(64,
               (3,3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same',
               name='Conv2')(x)
    x = layers.MaxPooling2D((2,2), name='pool2')(x)

    # We have used two max pool with pool size and strides of 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing it to RNNs
    new_shape = ((MAX_WIDTH // 4), (MAX_HIGHT // 4)*64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128,
                                         return_sequences=True,
                                         dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64,
                                         return_sequences=True,
                                         dropout=0.25))(x)

    # Predictions
    x = layers.Dense(len(ALPHABETS)+1,
              activation='softmax',
              name='dense2',
              kernel_initializer='he_normal')(x)

    # Calculate CTC
    output = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(inputs=[input_img,
                                       labels,
                                       input_length,
                                       label_length],
                                outputs=output,
                                name='ocr_model_v1')

    # Optimizer
    sgd = keras.optimizers.SGD(learning_rate=0.002,
                               momentum=0.9,
                               nesterov=True,
                               clipnorm=5)

    adam = keras.optimizers.Adam(learning_rate=0.001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 clipnorm=1)
    # Compile the model and return
    model.compile(optimizer=adam)
    print(model.summary())


    return model

print(1)
model = build_model()
print(2)
file_path_weights = '/mnt/c/dev/tmp20240710/C_LSTM_best.weights.h5'
print(3)
checkpoint = ModelCheckpoint(filepath=file_path_weights,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
print(4)
callbacks_list = [checkpoint,
                  PlotPredictions(frequency=1),
                  EarlyStopping(patience=8, verbose=1,monitor='val_loss', restore_best_weights=True)]
print(5)
# Train the model
#model.load_weights('/mnt/c/dev/tmp20240609/C_LSTM_first_Long_run_on_100k.best.weights.h5')
history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=validation_dataset,
                    verbose=1,
                    callbacks=callbacks_list,
                    shuffle=True)


print(6)


#model.load_weights('/mnt/c/dev/tmp20240609/C_LSTM_first_Long_run_on_100k.best.weights.h5')




score = model.evaluate(validation_dataset, verbose=0)
print('Score: ' +str(score))

###Build Prediction Modell
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
prediction_model.summary()












# A utility to decode the output of the network
def decode_batch_predictions(pred):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                        input_length=input_len,
                                        greedy=True)[0][0]

    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >=0:
                outstr += labels_to_char[c]
        output_text.append(outstr)

    # return final text results
    return output_text














imgT = []
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in validation_dataset.take(1):
    images = batch["image"]

    pred = prediction_model.predict(images)
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    results = keras.backend.ctc_decode(pred,
                                        input_length=input_len,
                                        greedy=True)[0][0]
    labels = batch["label"]
    len_labels_padded = batch["input_length"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = util.num_to_label(results[i])
        print(results[i])
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()