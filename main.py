from keras.models import load_model

from Dataset_Detection import *
from metrics import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

model = load_model('./Model/saved-model-34-0.2283-0.7823.h5',
                   custom_objects={'auprc': auprc, 'auprc0': auprc0, 'auprc1': auprc1, 'auprc2': auprc2,
                                   'auprc3': auprc3, 'auprc1to3': auprc1to3, 'f1m': f1m})

batch_size = 40
train_gen, validation_gen, test_gen = generators("/data/datasets/soccernet/SoccerNet-code/src", "ResNET", batch_size,
                                                 batch_size)

# scoresv = model.evaluate_generator(validation_gen, steps=(200 // batch_size))
scorest = model.evaluate_generator(test_gen, steps=(200 // batch_size))

# print(scoresv)
print(scorest)
