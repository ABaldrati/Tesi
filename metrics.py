from typing import Callable
from decorator import decorator
import tensorflow as tf
from keras import backend as K


@decorator
def metric(metric: Callable[[tf.Tensor, tf.Tensor], float], labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Wrap given metric for being used in Keras.
        metric:Callable[[tf.Tensor, tf.Tensor], float], metric to be wrapped.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    score = metric(labels, predictions)
    K.get_session().run(tf.local_variables_initializer())
    return score


@metric
def auprc(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels, predictions, curve='PR')[1]


@metric
def auprc0(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels[:, 0], predictions[:, 0], curve='PR')[1]


@metric
def auprc1(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels[:, 1], predictions[:, 1], curve='PR')[1]


@metric
def auprc2(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels[:, 2], predictions[:, 2], curve='PR')[1]


@metric
def auprc3(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels[:, 3], predictions[:, 3], curve='PR')[1]


@metric
def auprc1to3(labels: tf.Tensor, predictions: tf.Tensor) -> float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels[:, 1:4], predictions[:, 1:4], curve='PR')[1]


def f1m(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true[:, 1:4] * y_pred[:, 1:4], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, 1:4], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true[:, 1:4] * y_pred[:, 1:4], 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred[:, 1:4], 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


'''
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
    '''
