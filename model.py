from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop

from Dataset import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

train_gen, validation_gen, test_gen = generators("/home/alberto/Scrivania/SoccerNet-code-master/data", "ResNET", 50)

model = Sequential()
model.add(layers.LSTM(128,
                      activation='relu',
                      # dropout=0.1,
                      # recurrent_dropout=0.5,
                      return_sequences=True,
                      input_shape=(None, 512))
          )
model.add(layers.LSTM(256,
                      activation='relu',
                      # dropout=0.1,
                      # recurrent_dropout=0.5,
                      # input_shape=(None, 512)
                      )
          )
model.add(layers.Dense(4, activation='sigmoid'))

model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(
    train_gen,
    steps_per_epoch=12,
    epochs=20,
    validation_data=validation_gen,
    validation_steps=4,
    verbose=1,
)
