
from tensorflow.keras import layers
from tensorflow import keras
import util

MAX_HIGHT, MAX_WIDTH, LABELS_File, ALPHABETS, MAX_STR_LEN, NUM_OF_CHARACTERS, NUM_OF_TIMESTAMPS, BATCH_SIZE = util.get_global_var()

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
                            name='input_data',
                            dtype='float32')
    labels = layers.Input(name='input_label', shape=[MAX_STR_LEN], dtype='float32')
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



def build_deeper_model():
    input_img = layers.Input(shape=(MAX_WIDTH, MAX_HIGHT, 1), name='input_data', dtype='float32')
    labels = layers.Input(name='input_label', shape=[MAX_STR_LEN], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # First conv block with Batch Normalization
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1')(
        input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Second conv block with Batch Normalization
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    new_shape = ((MAX_WIDTH // 4), (MAX_HIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.25)(x)

    # RNNs with increased capacity
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(len(ALPHABETS) + 1, activation='softmax', name='dense2', kernel_initializer='he_normal')(x)

    output = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)

    model = keras.models.Model(inputs=[input_img, labels, input_length, label_length], outputs=output, name='ocr_model_v2')

    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1)

    model.compile(optimizer=adam)
    print(model.summary())

    return model


def residual_block(x, filters, kernel_size):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation=None, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_super_deep_model():
    input_img = layers.Input(shape=(MAX_WIDTH, MAX_HIGHT, 1), name='input_data', dtype='float32')
    labels = layers.Input(name='input_label', shape=[MAX_STR_LEN], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')

    # First conv block
    x = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv1')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), name='pool1')(x)

    # Second conv block
    x = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same', name='Conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), name='pool2')(x)

    # Additional Residual Blocks
    x = residual_block(x, 64, (3, 3))
    x = residual_block(x, 64, (3, 3))

    new_shape = ((MAX_WIDTH // 4), (MAX_HIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name='reshape')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.25)(x)

    # RNNs with increased capacity and Layer Normalization
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(len(ALPHABETS) + 1, activation='softmax', name='dense2', kernel_initializer='he_normal')(x)

    output = CTCLayer(name='ctc_loss')(labels, x, input_length, label_length)

    model = keras.models.Model(inputs=[input_img, labels, input_length, label_length], outputs=output, name='ocr_model_v3')

    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, clipnorm=1)

    model.compile(optimizer=adam)
    print(model.summary())

    return model