import keras.backend as K


class TrainingMetrics:
    def __init__(self, nb_classes, classes_weights):
        """  Metrics used for the training and/or monitoring of the model.

        Args:
            nb_classes: int, number of classes in the segmentation model
            classes_weights: a list of float of length = nb_classes, contains weights for each class
        """
        self.nb_classes = nb_classes
        self.classes_weights = classes_weights

    def _dice(self, y_true, y_pred, smooth=1.):
        """ Dice score.

        Args:
            y_true: keras_var, True value of the prediction
            y_pred: keras_var, Model prediction
            smooth: float, value added to avoid division by zero

        Returns:
            Dice score for y_true and y_pred.
        """
        flat_y_true = K.flatten(y_true)
        flat_y_pred = K.flatten(y_pred)
        # flat_y_true is a binary vector
        intersect = K.sum(flat_y_true * flat_y_pred)
        s_true = K.sum(flat_y_true)
        s_pred = K.sum(flat_y_pred)
        return (2. * intersect + smooth) / (s_true + s_pred + smooth)

    def prostate_dice(self, y_true, y_pred):
        """ Inner function to compute Dice for the prostate segmentation (2 classes problem).
        This method is also used as a monitoring metrics.

        Args:
            y_true: keras_var, True value of the prediction for the prostate segmentation
            y_pred: keras_var, Model prediction for the 2 classes problem

        Returns:
            The dice score for prostate segmentation.
        """
        prostate_dice = K.variable(0., name='prostate_dice')

        prostate_dice += self._dice(y_true[:, :, :, 1],
                                    y_pred[:, :, :, 1])

        return prostate_dice

    @staticmethod
    def _class_in_groundtruth(y_true_class):
        """ Check if a class is present in the ground truth. If the sum of the ground truth for this class is 0, then
        the class is not present. Otherwise, the class is present.

        Args:
            y_true_class: keras_var, True value of the prediction for a class

        Returns:
            keras_var, 0 if the class is not in ground truth, 1 otherwise
        """
        return K.clip(K.sum(K.flatten(y_true_class)), 0, 1)

    def _one_class_dice(self, y_true_class, y_pred_class):
        """Compute Dice for a specific class. If the class is not present in the ground truth, returns 0.

        Args
            y_true_class: keras_var, True value of the prediction for this class
            y_pred_class: keras_var, Model prediction for this class

        Returns:
            res, Dice score for y_true_class and y_pred_class
            class_in_gt_indicator, keras_var: 0 if the class is not in ground truth, 1 otherwise
        """
        class_in_gt_indicator = self._class_in_groundtruth(y_true_class)
        res = class_in_gt_indicator * self._dice(y_true_class, y_pred_class)

        return res, class_in_gt_indicator

    def _one_class_weighted_dice(self, y_true_class, y_pred_class, weight_class):
        """ Compute weighted Dice for a specific class. Each class is not necessarily present in the data. Here, the
        Dice is incremented only if the class is present in the ground truth.

        Args:
            y_true_class: keras_var, True value of the prediction for this class
            y_pred_class: keras_var, model prediction for this class
            weight_class: float, value to weight the class Dice with

        Returns:
            weighted_res, the weighted class Dice
            weights_sum, keras var: weight_class if the class is in the ground truth, 0 otherwise
        """
        res, class_in_gt_indicator = self._one_class_dice(y_true_class=y_true_class, y_pred_class=y_pred_class)
        weighted_res = res * weight_class
        weight_sum = class_in_gt_indicator * weight_class

        return weighted_res, weight_sum

    def weighted_classes_dice(self, y_true, y_pred):
        """ Inner function to compute Dice on each class.

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction.

        Returns:
            The weighted sum of "Dice" scores over all the classes.
        """
        res = K.variable(0., name='dice_classes')
        # Initialize with epsilon value to avoid division by zero
        weights_sum = K.epsilon()

        for i in range(self.nb_classes):
            res_class, weight_class = self._one_class_weighted_dice(
                y_true_class=y_true[:, :, :, i],
                y_pred_class=y_pred[:, :, :, i],
                weight_class=self.classes_weights[i]
            )
            res += res_class
            weights_sum += weight_class

        return res / weights_sum

    def dice_on_prostate(self, y_true, y_pred):
        """ Inner function to compute Dice on the prostate, i.e. on each class but background. Each class Dice is not
        weighted.

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction.

        Returns:
            The dice score over the prostate gland.
        """
        res = K.variable(0., name='dice_classes')
        # Initialize with epsilon value to avoid division by zero
        class_sum = K.epsilon()

        for i in range(1, self.nb_classes):
            # Each class is not necessarily present in the data. Compute Dice only for classes
            # which are represented in the ground truth.
            class_res, class_in_gt_indicator = self._one_class_dice(
                y_true_class=y_true[:, :, :, i],
                y_pred_class=y_pred[:, :, :, i]
            )
            res += class_res
            class_sum += class_in_gt_indicator

        res /= class_sum

        return res

    def lesion_dice(self, y_true, y_pred):
        """ Inner function to compute the Dice on the lesions classes, thus ignoring the prostate segmentation.

        Args:
            y_true: keras_var, True value of the prediction.
            y_pred: keras_var, Model prediction.

        Returns:
            The lesion Dice score.
        """
        res = K.variable(0., name='lesion_dice')
        # Initialize with epsilon value to avoid division by zero
        class_sum = K.epsilon()

        for i in range(2, self.nb_classes):
            class_res, class_in_gt_indicator = self._one_class_dice(
                y_true_class=y_true[:, :, :, i],
                y_pred_class=y_pred[:, :, :, i]
            )
            res += class_res
            class_sum += class_in_gt_indicator

        res /= class_sum

        return res
