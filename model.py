from extra_keras_metrics import auprc
from keras import backend as K
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop

from Dataset import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


train_gen, validation_gen, test_gen = generators("/home/alberto/Scrivania/SoccerNet-code-master/data", "ResNET", 50)

model = Sequential()
model.add(layers.GRU(128,
                     activation='relu',
                     # dropout=0.1,
                     # recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, 512))
          )
model.add(layers.GRU(256,
                     activation='relu',
                     # dropout=0.1,
                     # recurrent_dropout=0.5,
                     # input_shape=(None, 512)
                     )
          )
model.add(layers.Dense(4, activation='sigmoid'))

model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=[auprc])

history = model.fit_generator(
    train_gen,
    steps_per_epoch=12,
    epochs=20,
    validation_data=validation_gen,
    validation_steps=4,
    verbose=1,
)
