from contextlib import redirect_stdout
import keras
import matplotlib.pyplot as plt
from keras import layers, optimizers
from keras.models import Sequential
from Dataset import *
from metrics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

'''if not os.path.exists('./Model/my_model_log_dir'):
    os.mkdir('./Model/my_model_log_dir')
'''

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_f1m',
        patience=20,
        mode='max',
        # min_delta=-0.0001
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='./Model/saved-model-{epoch:02d}-{val_loss:.4f}-{val_f1m:.4f}.h5',
        monitor='val_loss',
        # mode='min',
        save_best_only=False
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1m',
        factor=0.4,
        patience=10,
        mode='max',
        min_lr=0.00005,
        # min_delta=-0.0001
    ),
    keras.callbacks.CSVLogger(
        filename='./Model/my_model.csv',
        separator=',',
        append=True
    ),
]

batch_size_train = 15
batch_size_val_and_test = 20
train_gen, validation_gen, test_gen = generators("/data/datasets/soccernet/SoccerNet-code/src", "ResNET",
                                                 batch_size_train, batch_size_val_and_test, False)

model = Sequential()
model.add(layers.Bidirectional(layers.GRU(512,
                                          activation='relu',
                                          dropout=0.1,
                                          recurrent_dropout=0.4,
                                          return_sequences=True,
                                          # kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001)
                                          ),
                               input_shape=(None, 512))
          )

model.add(layers.Bidirectional(layers.GRU(256,
                                          activation='relu',
                                          dropout=0.1,
                                          recurrent_dropout=0.4,
                                          return_sequences=True,
                                          # kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001)
                                          )
                               )
          )

model.add(layers.Bidirectional(layers.GRU(128,
                                          activation='relu',
                                          dropout=0.1,
                                          recurrent_dropout=0.4,
                                          # kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001)
                                          )
                               )
          )

model.add(layers.Dense(4,
                       activation='sigmoid')
          )

model.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy',
              metrics=[auprc, auprc0, auprc1, auprc2, auprc3, auprc1to3, keras.metrics.binary_accuracy, f1m])

history = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=(600 // batch_size_train),
    epochs=152,
    validation_data=validation_gen,
    validation_steps=(200 // batch_size_val_and_test),
    verbose=1,
    callbacks=callbacks_list,
    shuffle=False,
    # initial_epoch=32,
)

'''
scoresv = model.evaluate_generator(validation_gen, steps=(200 // batch_size))
scorest = model.evaluate_generator(test_gen, steps=(200 // batch_size))

print(scorest)
print(scoresv)
'''
with open('./Model/mymodelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./Model/training-validation-loss')

plt.clf()
mAP = history.history['f1m']
val_mAP = history.history['val_f1m']
plt.plot(epochs, mAP, 'bo', label='Training f1m')
plt.plot(epochs, val_mAP, 'b', label='Validation f1m')
plt.title('Training and validation f1m  ')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.savefig('./Model/training-validation-f1m')

# test_loss, auprc, auprc0, auprc1, auprc2, auprc3, auprc1to3 = model.evaluate_generator(test_gen, steps=10)
# print('test acc:', test_acc)

