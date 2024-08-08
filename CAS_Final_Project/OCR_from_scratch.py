import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import models
import util
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


tf.config.run_functions_eagerly(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#reading all key values from the folder on the drive
keyValMNIST, keyValch4, keyValBd, keyVal100k, keyValIdVehic, keyValSVHN = util.load_all_key_val()


key_val = np.concatenate((keyValMNIST, keyValBd), axis=0)
key_val = util.load_key_val_syntEVN()

# preprocessing all images that are in the key_val and convert them into the 4 Vectors required for ctc loss
x_all_img_total_path, labels_padded, len_labels_padded, len_labels_not_padded = util.process_key_values_into_ctc_requierd_attributes(key_val)

# Creating tensorflow datasets. Dataset is foreseen for a ctc loss network
train_dataset, validation_dataset = util.create_tensorflow_train_and_validation_dataset(x_all_img_total_path, labels_padded, len_labels_padded, len_labels_not_padded, split_ratio = 0.8)



_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["input_data"]
    labels = batch["input_label"]
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
        self.batch_images = list(batch.as_numpy_iterator())[0]["input_data"]
        self.batch_labels = list(batch.as_numpy_iterator())[0]["input_label"]

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



#########################################################
#################Getting the Model#######################
#########################################################
model = models.build_super_deep_model()


file_path_weights = '/mnt/c/dev/tmp20240710/C_LSTM_best.weights.h5'
checkpoint = ModelCheckpoint(filepath=file_path_weights,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')

callbacks_list = [checkpoint,
                  PlotPredictions(frequency=1),
                  EarlyStopping(patience=80, verbose=1,monitor='val_loss', restore_best_weights=True)]


# Train the model with pretrained weights
#model.load_weights('/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/Weights/100k_batch256_alphaAll.weights.h5')
history = model.fit(train_dataset,
                    epochs=2500,
                    validation_data=validation_dataset,
                    verbose=1,
                    callbacks=callbacks_list,
                    shuffle=True)



score = model.evaluate(validation_dataset, verbose=0)
print('Score: ' +str(score))

###Build Prediction Modell
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
prediction_model.summary()




imgT = []
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in validation_dataset.take(1):
    images = batch["input_data"]

    pred = prediction_model.predict(images)
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0])*pred.shape[1]
    results = keras.backend.ctc_decode(pred,
                                        input_length=input_len,
                                        greedy=True)[0][0]
    labels = batch["input_label"]
    len_labels_padded = batch["input_length"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = util.num_to_label(results[i])
        #print(results[i])
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()