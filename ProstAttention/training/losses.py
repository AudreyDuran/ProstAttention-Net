import keras.backend as K
import numpy as np
from keras import losses as kl

from ProstAttention.training.training_metrics import TrainingMetrics
from ProstAttention.training.training_utils import reduce_crossentropy


class Losses:

    def __init__(self, nb_classes, classes_weights):
        """  Losses to train the model

        Args:
            nb_classes: int, Number of classes in the segmentation model
            classes_weights: a list of float of length = nb_classes, weights for each class
        """
        self.nb_classes = nb_classes
        self.classes_weights = [1.] * nb_classes if not classes_weights else classes_weights
        self.metrics = TrainingMetrics(nb_classes=self.nb_classes, classes_weights=self.classes_weights)

    def prostate_dice_loss(self, y_true, y_pred):
        """ Compute the loss for the prostate segmentation problem : (1 - prostate_dice)

        Args:
            y_true: keras_var, True value of the prediction with self.nb_classes classes
            y_pred: keras_var, Model prediction (probabilities) for the 2 classes problem

        Returns:
            The prostate dice loss.
        """
        return 1 - self.metrics.prostate_dice(y_true, y_pred)

    def prostate_combined_loss(self, y_true, y_pred):
        """ Compute the loss : crossentropy loss + prostate dice loss. Crossentropy is computed for both the prostate
        class and background. Dice is only computed for the prostate class.

        Args:
            y_true: keras_var, True value of the prediction for prostate segmentation
            y_pred: keras_var, Model prediction (probabilities) for prostate segmentation

        Returns:
           The combined loss.
        """
        c_loss = kl.categorical_crossentropy(y_true, y_pred)
        d_loss = self.prostate_dice_loss(y_true, y_pred)

        return c_loss + d_loss

    def classes_crossentropy(self, y_true, y_pred):
        """ Inner function to compute the crossentropy on each class.

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction (probabilities).

        Returns:
            The crossentropy of each pixel weighted by a mask.
        """
        masking_classes_weights = np.array(self.classes_weights)[None, None, None, :]
        mask = K.sum(y_true * masking_classes_weights, axis=3)

        res = kl.categorical_crossentropy(y_true, y_pred)
        return res * mask

    def weighted_classes_dice_loss(self, y_true, y_pred):
        """ Compute the loss: 1 - dice(y_true, y_pred)

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction (probabilities).

        Returns:
            The classes-weighted dice loss
        """
        return 1 - self.metrics.weighted_classes_dice(y_true, y_pred)

    def combined_loss(self, y_true, y_pred):
        """ Compute the loss : crossentropy loss + dice loss

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction (probabilities).

        Returns:
            The combined loss.
        """
        c_loss = self.classes_crossentropy(y_true, y_pred)
        c_loss = reduce_crossentropy(c_loss)
        d_loss = self.weighted_classes_dice_loss(y_true, y_pred)

        return c_loss + d_loss
