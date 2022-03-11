import json
import os
import warnings
from glob import glob
from os.path import join as pjoin
from re import findall

import sklearn

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)
import h5py
import keras.backend as K
import numpy as np
from VITALabAI.dataset.semanticsegmentation.acdc.preprocess import PreProcessRescaleIntensity
from VITALabAI.VITALabAiKerasAbstract import VITALabAiKerasAbstract
from VITALabAI.utils.export import save_nii_file
from VITALabAI.utils.utils import print_confusion_matrix, create_model_directories
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from ProstAttention.dataset.generator import Dataset
from ProstAttention.dataset.utils import apply_argmax_to_set
from ProstAttention.training.callbacks import LossWeightsCallback
from ProstAttention.training.losses import Losses


class BaseSegmentationModel(VITALabAiKerasAbstract):
    """Base model for a segmentation task on our Dataset class"""

    def __init__(self, dataset: Dataset, optimizer=None, nb_feature_maps=32, l2reg=1e-4, name=None,
                 pretrained_model=None, class_weights=None, target_loss_weight=None, loss_update_epoch=None,
                 patience=50, nb_epochs=200):
        """
        Args:
            dataset: Dataset instance, the dataset to train the model with
            optimizer: Optimizer to use
            nb_feature_maps: int, number of feature maps at beginning of the network, automatic scaling for the
             following layers
            l2reg: float, alpha in the regularizer
            name: string, name of the model
            pretrained_model: str, path to model weights
            class_weights: a list of float of length = nb_classes, contains weights for each classes
            target_loss_weight: float, value that the lesion segmentation loss weight should be changed to in
              LossWeightsCallback in case of a 2 output branch model
            loss_update_epoch: int, epoch on which the weights on losses should be changed (with a Callback)
            patience: int, number of epochs with no improvement used by early_stopping and reduce_lr
            nb_epochs: int, number of epoch to train the model on the dataset
        """
        # Set these parameters before super.__init__ because they are needed to build the model.
        self.nb_feature_maps = nb_feature_maps
        self.l2reg = l2reg
        self.model_losses = Losses(dataset.nb_classes, class_weights)
        self.pretrained_model = pretrained_model
        self.target_loss_weight = target_loss_weight
        self.loss_update_epoch = loss_update_epoch
        self.patience = patience
        self.nb_epochs = nb_epochs

        super().__init__(dataset, self.losses(), optimizer, self.metrics(), loss_weights=self.losses_weights())

        self.name = self.__class__.__name__ if name is None else name
        self.name, self.checkpoint_dir, self.logs_dir, self.model_dir = \
            create_model_directories(self.name, ["ModelCheckpoint", "LOGS", "MODEL"])
        self.results_dir = pjoin(self.name, "RESULTS")
        self.results_dir, self.conf_mat_dir, self.classif_report_dir = \
            create_model_directories(self.results_dir, ["ConfusionMatrices", "ClassificationReports"])

        with open(pjoin(self.name, "run_conf.json"), "w") as fj:
            json.dump(self.get_config_dict(), fj, indent=2)

    @staticmethod
    def get_preprocessing():
        return PreProcessRescaleIntensity(),

    def losses(self, **kwargs):
        raise NotImplementedError("Abstract class")

    def losses_weights(self):
        return None

    def metrics(self, **kwargs):
        """ Metrics used during the training.
        """
        raise NotImplementedError("Abstract class")

    def build_model(self):
        raise NotImplementedError("Abstract class")

    def save(self):
        """Method to save the best weights. """
        # Extract the best weights
        list_of_files = glob(pjoin(self.checkpoint_dir, '*'))
        # Check that there is at least one weights file
        if len(list_of_files) != 0:
            max_dice = max(list_of_files, key=self.extract_monitored_value)
            print('Best weights saving ... \n' + max_dice)

            # Name the file
            h5_name = f'{self.__class__.__name__}_input{self.dataset.input_size}_{self.nb_feature_maps}fm_' \
                      f'epoch{self.extract_epoch(max_dice)}.h5'

            # Load weights and save in the right folder
            self.load_weights(max_dice)
            self.model.save(pjoin(self.model_dir, h5_name))
        else:
            print('No weights available for best weights saving.')

    @staticmethod
    def extract_monitored_value(path_to_file):
        """ Method to extract the monitored value (ie loss or dice) from a file name."""
        base = os.path.basename(path_to_file)
        base = os.path.splitext(base)[0]
        return float(findall("\d+\.\d+", base)[0])

    @staticmethod
    def extract_epoch(path_to_file):
        """ Method to extract the epoch number from a file name"""
        base = os.path.basename(path_to_file)
        base = os.path.splitext(base)[0]
        return base.split('_')[1]

    def predict_and_save(self, sequence, h5f, data_set):
        """ Function to predict and save results.

        Args:
            sequence: Keras.Sequence, sequence used to generate predictions
            h5f: h5py file to save predictions in
            data_set: string, the set corresponding to the input sequence ('train', 'validation', 'test')
        """
        gt_list = []
        pred_list = []
        for i in tqdm(range(len(sequence))):
            data = sequence[i]
            name, img, gt, v_size = data
            pred = self.model.predict_on_batch(img)
            group = h5f.create_group(name)
            self.save_to_h5f(group, img, gt, pred, v_size)
            gt_list.append(gt)
            pred_list.append(pred)

        # Parse over each pair of gt and predictions
        for i in range(2):
            gt = [gt_list[n][i] for n in range(len(gt_list))]
            pred = [pred_list[n][i] for n in range(len(pred_list))]
            gt_array = apply_argmax_to_set(np.array(gt))
            pred_array = apply_argmax_to_set(np.array(pred))

            self.save_classification_report(data_set, gt_array, pred_array, gt_pos=i)
            self.save_confusion_matrix(data_set, gt_array, pred_array, gt_pos=i)

    def evaluate(self, **kwargs):
        """Evaluate the model and save the results in hdf5 files for each available set ('test' set might not be
        present).

        Args:
            **kwargs: additional parameters
        """
        for set_name, set_prediction_file in self.dataset.prediction_filenames.items():
            print(f"Saving {set_name} set predictions...")
            with h5py.File(pjoin(self.results_dir, f'{set_prediction_file}'), "w") as h5f:
                self.predict_and_save(self.dataset.get_set_for_prediction(f'{set_name}'), h5f, f'{set_name}')

    def save_to_h5f(self, group, img, gt, pred, v_size):
        """ Save an image, gt, prediction and corresponding attributes in an HDF5 group.

        Args:
            group: hdf5 group
            img: array, input image
            gt: array, ground truth for the input image
            pred: array, model prediction for the input image
            v_size: list, voxel dimensions
        """
        group.create_dataset("image", data=img, compression='gzip')

        group.create_dataset("gt_prostate_c", data=gt[0], compression='gzip')
        group.create_dataset("gt_c", data=gt[1], compression='gzip')
        group.create_dataset("pred_prostate_c", data=pred[0], compression='gzip')
        group.create_dataset("pred_c", data=pred[1], compression='gzip')

        gts = [agt.argmax(axis=-1) for agt in gt]
        preds = [apred.argmax(axis=-1) for apred in pred]
        group.create_dataset("gt_prostate_m", data=gts[0], compression='gzip')
        group.create_dataset("pred_prostate_m", data=preds[0], compression='gzip')
        group.create_dataset("gt_m", data=gts[1], compression='gzip')
        group.create_dataset("pred_m", data=preds[1], compression='gzip')

        group.attrs['voxel_size'] = v_size

    def export_results(self):
        """This method exports results from the HDF5 files to nifti format for each set."""
        print("Exporting results...")
        nii_images = pjoin(self.results_dir, "NIFTI")
        if not os.path.isdir(nii_images):
            os.makedirs(nii_images)

        for which_set, dataset_filename in self.dataset.prediction_filenames.items():
            with h5py.File(pjoin(self.results_dir, dataset_filename), "r") as h5f:
                self.save_nifti(nii_images, which_set, h5f)

    def save_nifti(self, path, data_set, h5file):
        """This method loads the data from a hdf5 file and saves it to nifti format.

        Args:
            path: string, path to save the nifti files to.
            data_set: string, set being saved to nifti.
            h5file: h5py File, file to be loaded.
        """
        save_images = pjoin(path, data_set)
        if not os.path.isdir(save_images):
            os.makedirs(save_images)

        for patient_name in tqdm(h5file.keys(), desc=f"Save {data_set} nifti"):
            # Each of these names 'pred_m', ... are defined during
            # the prediction task. We only get the results of the useful ones
            # and save into nifti format.
            save_path = pjoin(save_images, f"{patient_name}")

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            pred_main = h5file[f"{patient_name}/pred_m"][()].squeeze()
            ground_main = h5file[f"{patient_name}/gt_m"][()].squeeze()

            def transpose_mask(mask):
                # We save it in "height, width, slices" (instead of "slices, height, width")
                if len(mask.shape) < 3:
                    mask = mask[np.newaxis]
                mask = mask.transpose((1, 2, 0))
                return mask

            pred_main = transpose_mask(pred_main)
            ground_main = transpose_mask(ground_main)

            img = h5file[f"{patient_name}/image"][()].squeeze()
            voxel = h5file[patient_name].attrs.get('voxel_size', default=[1, 1, 1])

            if len(img.shape) < 4:
                img = img[:, :, :, np.newaxis]

            channels = self.dataset.channels
            img = img.transpose((1, 2, 0, 3))
            # Split each channel into its own 3D image and save it in a list
            img_channels = [img[:, :, :, i] for i in range(len(channels))]

            datas = [pred_main, ground_main] + img_channels
            tags = ["prediction", "groundtruth"] + [f"image-ch{i}" for i in channels]

            pred_prostate = h5file[f"{patient_name}/pred_prostate_m"][()].squeeze()
            gt_prostate = h5file[f"{patient_name}/gt_prostate_m"][()].squeeze()
            pred_prostate = transpose_mask(pred_prostate)
            gt_prostate = transpose_mask(gt_prostate)
            datas += [pred_prostate, gt_prostate]
            tags += ["prediction_prostate", "groundtruth_prostate"]

            for data, tag in zip(datas, tags):
                save_nii_file(data,
                              pjoin(save_path, f"{patient_name}_{tag}.nii.gz"),
                              zoom=voxel)

    def save_confusion_matrix(self, data_set, gt, pred, gt_pos=None):
        """This method saves a confusion matrix figure for the specified set. It uses the sklearn function :
        sklearn.metrics.confusion_matrix.

        Args:
            data_set: string, which set confusion matrix is being saved ('train', 'validation', 'test')
            gt: array, ground truth for the input image of shape (nb_patients, nb_slices, W, H)
            pred: array, model prediction for the input image of shape (nb_patients, nb_slices, W, H)
            gt_pos: int, position of the concerned ground-truth in case of several ground-truth (ie. several outputs)
        """
        # Remove labels which are not in the data for the specified data set
        labels = np.unique(gt)

        # Compute confusion matrix figure and save it in the results folder
        conf_mat = confusion_matrix(y_true=gt, y_pred=pred, labels=labels)
        conf_fig = print_confusion_matrix(conf_mat, class_names=labels)
        save_path = pjoin(self.conf_mat_dir, f"confusion_mat_{data_set}{gt_pos}.png")
        conf_fig.savefig(save_path)

        # Compute and save normalized confusion matrix
        conf_fig = print_confusion_matrix(conf_mat, normalize=True, class_names=labels)
        save_path = pjoin(self.conf_mat_dir, f"normalized_confusion_mat_{data_set}{gt_pos}.png")
        conf_fig.savefig(save_path)

    def save_classification_report(self, data_set, gt, pred, gt_pos=None):
        """This method saves the classification report showing the main classification metrics for the specified set,
         from report returned by sklearn.metrics.classification_report.

        Args:
            data_set: string, which set confusion matrix is being saved ('train', 'validation', 'test')
            gt: array, ground truth for the input image of shape (nb_patients, nb_slices, W, H)
            pred: array, model prediction for the input image of shape (nb_patients, nb_slices, W, H)
            gt_pos: int, position of the concerned ground-truth in case of several ground-truth (ie. several outputs)
        """
        report = classification_report(y_true=gt, y_pred=pred, labels=np.arange(self.dataset.nb_classes))

        with open(pjoin(self.classif_report_dir, f'classification_report_{data_set}{gt_pos}.csv'),
                  'w') as csv_file:
            csv_file.write(report)

    def get_model_monitor(self):
        """This method returns the quantity to monitor for early stopping and to decide the best model weights.

        Returns:
            str, the quantity to monitor
        """
        return 'val_lesion_seg_dice_on_prostate'

    def get_callbacks(self, learning_rate, early_stopping=True, reduce_lr=False, update_loss_weights=False,
                      patience=10, model_monitor="val_dice_on_prostate", target_weight=1., update_epoch=20):
        """This method returns the callbacks used during the models training.

        Args:
            learning_rate: float, learning rate, used to specify min_lr
            early_stopping: bool, if True early stopping is used
            reduce_lr: bool, if True, reduce learning on plateau is used
            update_loss_weights: bool, if True, begin with a loss weights = [1, 0] and changed to equal weights at a
                given epoch (update_epoch)
            patience: int, number of epochs with no improvement used by early_stopping and reduce_lr
            model_monitor: str, name of the quantity to monitor for ModelCheckpoint and EarlyStopping callbacks
            target_weight: float, weight to assign to the lesion segmentation branch loss when the specified epoch
                is reached
            update_epoch: int, epoch on which the weights on losses should be changed

        Returns:
            list of callbacks
        """
        # Create callback for TensorBoard
        tensorboard = TensorBoard(log_dir=self.logs_dir, histogram_freq=0, write_graph=True,
                                  write_images=False)
        # Create model saving callback
        details = self.__class__.__name__
        details += '_{epoch:03d}_{' + model_monitor + ':.4f}.h5'

        model_saver = ModelCheckpoint(
            pjoin(self.name, "ModelCheckpoint", details),
            monitor=model_monitor,
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )

        terminate_on_nan = TerminateOnNaN()

        csvlogger = CSVLogger(pjoin(self.name, 'csvlog.csv'), separator=',', append=False)
        early_stopping_callback = EarlyStopping(monitor=model_monitor, patience=patience, verbose=True, mode='max')

        min_lr = learning_rate / 20
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=True, patience=patience // 2,
                                               min_lr=min_lr)

        list_callback = [tensorboard, model_saver, csvlogger, terminate_on_nan]

        if early_stopping:
            list_callback.append(early_stopping_callback)

        if reduce_lr:
            list_callback.append(reduce_lr_callback)

        if update_loss_weights:
            list_callback.append(
                LossWeightsCallback(output_branch='lesion_seg', update_epoch=update_epoch, target_weight=target_weight))

        return list_callback

    def get_config_dict(self):
        """ Returns dictionary with different parameters of the model.

        Returns:
            dict, parameters of the model
        """
        if type(self.model.loss) == list:
            losses = [l.__name__ for l in self.model.loss]
        else:
            losses = [l.__name__ for l in self.model.loss.values()]

        run_conf = {
            "model": {
                "name": os.path.realpath(self.__class__.__name__),
                "attention_layers": self.layer_attention if hasattr(self, "layer_attention") else None,
                "nb_feature_maps": self.nb_feature_maps,
                "regularization": self.l2reg,
                "pretrained_model_file": self.pretrained_model
            },
            "dataset": {
                "name": self.dataset.__class__.__name__,
                "path": self.dataset.path if hasattr(self.dataset, "path") else None,
                "data_tag": self.dataset.final_gt_tag,
                "prostate_data_tag": self.dataset.prostate_gt_tag if hasattr(self.dataset, "prostate_gt_tag") else None,
                "nb_classes": self.dataset.nb_classes,
                "channels": self.dataset.channels,
                "batch_size": self.dataset.batch_size,
                "input_size": self.dataset.input_size,
            },
            "training": {
                "losses": losses,
                "initial_losses_weights": self.losses_weights(),
                "target_lesion_loss_weight": self.target_loss_weight,
                "nb_epochs": self.nb_epochs,
                "update_epoch": self.loss_update_epoch,
                "classes_weights": self.model_losses.classes_weights,
                "monitored_metric": self.get_model_monitor(),
                "patience": self.patience,
                "optimizer": {
                    "name": self.model.optimizer.__class__.__name__,
                    "initial_lr": str(K.eval(self.model.optimizer.lr)),
                },
            },
            "pre_processing": [p.__class__.__name__ for p in self.get_preprocessing()],
        }
        return run_conf

    @staticmethod
    def build_generic_parser():
        """ Sets-up the parser to handle generic options common to any model.

        Returns:
            the parser object that supports generic options common to any model.
        """

        import argparse

        aparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        aparser.add_argument("path", help="Path to the hdf5 dataset", type=str)
        aparser.add_argument("--classes", help="Number of classes in the segmentation model. It should be the same "
                                               "value as the one specified in generate_hdf5.py", type=int, default=6)
        aparser.add_argument("--channels", type=int, nargs='+', choices=[0, 1], default=[0, 1],
                             help="Channels to use in the input. 0 includes T2-w, 1 includes ADC. "
                                  "Both are used by default.")
        aparser.add_argument("--crop", help="Size of the input images after cropping. The crop value must be divisible"
                                            " by the 2**5", type=int, default=96)
        aparser.add_argument("--batch_size", type=int, help="choose the size of the batch feeded to the network.",
                             default=32)
        aparser.add_argument("--name", type=str, dest="name", help="Model path", default=None)
        aparser.add_argument("--lr", type=float, dest="lr", help="learning rate", default=1e-3)
        aparser.add_argument("--reg", type=float, dest="reg", help="regularization strength", default=1e-4)
        aparser.add_argument("--nb_feature_maps", type=int, help="Number of features maps used on the first "
                                                                 "layer", default=32)
        aparser.add_argument("--workers", type=int, help="# of workers to use for training", default=1)
        aparser.add_argument("--pretrained_model", type=str, help="Model h5 file (as contained in the MODEL folder).",
                             default=None)
        aparser.add_argument("--testonly", action="store_true", dest="testonly", help="Skip training and do test phase")
        aparser.add_argument("--nb_epochs", type=int, help="Number of epoch to train on the dataset", default=200)
        aparser.add_argument("--patience", type=int, dest="patience", help="Number of epoch without loss improvement "
                                                                           "before stopping training",
                             default=50)
        aparser.add_argument("--no_earlystopping", dest="early_stopping", action='store_false',
                             help="Disable early stopping. Early stopping is enabled by default.")
        aparser.add_argument("--class_weights", type=float, nargs='+', help="Weight for each class, used in loss "
                                                                            "functions and metrics.",
                             default=[0.002, 0.14, 0.1715, 0.1715, 0.1715, 0.1715])
        aparser.add_argument("--subfold", help="The index of the validation subfold for this experiment (used only in "
                                               "case of cross-validation experiment)", type=int,
                             default=None, choices=range(10))
        aparser.add_argument("--no_hdf5_load", dest="load_hdf5", help="Doesn't load the .hdf5 into the RAM",
                             action="store_false")
        return aparser
