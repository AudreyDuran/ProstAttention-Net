import warnings
from os.path import join as pjoin, basename

warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
import numpy as np
from VITALabAI.utils.dataset import centered_resize
from keras.utils import Sequence
from ProstAttention.project_utils.utils import flatten


class Dataset:
    """ Wrapper for the dataset.
    """
    TRAIN_KEY = 'train'
    VAL_KEY = 'validation'
    TEST_KEY = 'test'

    def __init__(self, path, nb_classes, channels, crop_size: int = 96, batch_size=4, seed=None,
                 preproc_fn=(), subfold_index=None, final_gt_tag="gt", prostate_gt_tag="gt_prostate", load_hdf5=True):
        """
        Args:
            path: string, path to the hdf5 file
            nb_classes: int, number of classes
            channels: list of int, input channels to use. 0 includes T2-w, 1 includes ADC. Both modality can be used
                at the same time (2 input channels).
            crop_size: int, size of the image after cropping (for both width and height) in pixels
            batch_size: int, size of the batches to return
            seed: object, seed for the permutations
            preproc_fn: tuple, preprocessing functions to be applied on the data
            subfold_index: int, the validation subfold index in case of a cross-validation experiment
            final_gt_tag: str, the data tag to use in the hdf5 file for the final ground truth
            prostate_gt_tag: str, the data tag to use in the hdf5 file for the prostate ground truth
            load_hdf5: bool, if True load the .hdf5 file into the RAM for a much faster training (if the dataset size
             and GPU allows it)
        """
        self.input_size = crop_size
        self.batch_size = batch_size
        self.path = path
        self.nb_classes = nb_classes
        self.channels = channels
        self.preproc_fn = preproc_fn
        self.final_gt_tag = final_gt_tag
        self.prostate_gt_tag = prostate_gt_tag
        self.subfold_index = subfold_index

        np.random.seed(seed)

        self.loaded_dataset = {} if load_hdf5 else None
        self.train_list, self.val_list, self.test_list = self.load_data()
        self.prediction_filenames = {self.TRAIN_KEY: "TRAIN_PREDICTION", self.VAL_KEY: "VALID_PREDICTION"}
        # Add the test set if it is present in the dataset
        if len(self.test_list) > 0:
            self.prediction_filenames[self.TEST_KEY] = "TEST_PREDICTION"

    def get_train_set(self):
        """ Get the training set

        Returns:
            keras.project_utils.Sequence
        """

        return DataSequence(self.train_list, self.path, self.input_size, self.channels, self.batch_size,
                            preproc_fn=self.preproc_fn, prostate_gt_tag=self.prostate_gt_tag,
                            final_gt_tag=self.final_gt_tag, loaded_dataset=self.loaded_dataset)

    def get_validation_set(self):
        """ Get the training set

        Returns:
            keras.project_utils.Sequence
        """

        return DataSequence(self.val_list, self.path, self.input_size, self.channels, self.batch_size,
                            preproc_fn=self.preproc_fn, prostate_gt_tag=self.prostate_gt_tag,
                            final_gt_tag=self.final_gt_tag, loaded_dataset=self.loaded_dataset)

    def get_test_set(self):
        """ Get the training set

        Returns:
            keras.project_utils.Sequence
        """

        return DataSequence(self.test_list, self.path, self.input_size, self.channels, self.batch_size,
                            preproc_fn=self.preproc_fn, prostate_gt_tag=self.prostate_gt_tag,
                            final_gt_tag=self.final_gt_tag, loaded_dataset=self.loaded_dataset)

    def get_set_for_prediction(self, set):
        return TestDataSequence(set, self.path, self.input_size, self.channels, self.preproc_fn,
                                subfold_index=self.subfold_index)

    def get_input_shape(self):
        """Get the input shape of the dataset

        Returns:
            Tuple, shape of an input (H,W,C)
        """
        return self.input_size, self.input_size, len(self.channels)

    def load_data(self):
        """ Read list of images from the HDF5 file and save them in 3 set lists.

        Returns:
            tuple of list containing path to files for each set
        """

        def get_set_list(set_keys, file):
            set_list = []
            # In case of a cross-validation experiment, several subfold keys are used for the train set
            for set_key in set_keys:
                f_set = file[set_key]
                for key in list(f_set.keys()):
                    patient = f_set[key]
                    img = patient['img']
                    gt = patient[self.final_gt_tag]

                    assert img.shape[0] == gt.shape[0], "The number of slices must match"

                    gt_prostate = patient[self.prostate_gt_tag]
                    assert img.shape[0] == gt_prostate.shape[0], "The number of slices must match"

                    # For each slices
                    for i in range(img.shape[0]):
                        k = f'{set_key}/{key}'
                        set_list.append((k, i))
            return set_list

        with h5py.File(self.path, 'r') as f:
            # If it is a cross-validation experiment, subfold_index is not None
            if self.subfold_index is not None:
                train_keys = load_subfolds_in_set(set_name=self.TRAIN_KEY, file=f, subfold_index=self.subfold_index)
                validation_keys = load_subfolds_in_set(set_name=self.VAL_KEY, file=f, subfold_index=self.subfold_index)
            else:
                validation_keys = [self.VAL_KEY]
                train_keys = [self.TRAIN_KEY]

            train_list = get_set_list(train_keys, f)
            train_list = np.random.permutation(train_list)

            val_list = get_set_list(validation_keys, f)
            # The test set might not be present (e.g. in a cross-validation dataset)
            test_list = get_set_list([self.TEST_KEY], f) if self.TEST_KEY in f else []

        return train_list, val_list, test_list

    def view_sample(self, batch=0, sample=0, channels=[0, 1]):
        """ Plot a sample of the training set.

        Args:
            batch : int, index of the batch in which a sample will be plotted, default : 0
            sample : int, index of the sample to plot in the batch, default : 0
            channels: list, list of channels to show. (ex [0] for T2w, [0, 1] for T2w and ADC, etc.)
        """

        sequence = self.get_train_set()  # A sequence (random in train but not in test set)
        print("Plot image", self.train_list[batch * self.batch_size + sample])
        img, gts = sequence[batch]

        # Squeeze the first dimension of the array (batch_size=1)
        img = img.squeeze(axis=0)

        gts = [gt.squeeze() for gt in gts]
        # Before calling argmax, ground truth have a shape of (width, height, num_classes).
        # Final segmentation maps have a shape of (width, height)
        segs = [np.argmax(gt, axis=-1) for gt in gts]

        if len(img.shape) < 3:
            img = img[..., np.newaxis]

        for i in range(len(channels)):
            # Choose the final segmentation map to show (with lesion contours)
            _, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img[:, :, i], cmap='gray')
            ax2.imshow(segs[1])
            plt.show(block=False)

            plt.figure(3)
            plt.imshow(img[:, :, i], cmap='gray')
            plt.imshow(segs[1], alpha=0.2)
            plt.show()


class DataSequence(Sequence):
    """ Sequence of images for Keras."""

    def __init__(self, key_list, file_path, input_size, channels, batch_size,
                 preproc_fn=(), final_gt_tag="gt",
                 prostate_gt_tag=None, loaded_dataset=None):
        """
        Args:
            key_list: list, list of pairs of keys and indexes for each slice
            file_path: string, path to the hdf5 dataset file
            input_size: int, size of the image (for both width and height) in pixels
            channels: list of int, input channels to use. 0 includes T2-w, 1 includes ADC. Both modality can be used
                at the same time (2 input channels).
            batch_size: int, batch size to stack the images
            preproc_fn: tuple, preprocessing functions to be applied on the data
            final_gt_tag: str, the data tag to use in the hdf5 file for the final ground truth
            prostate_gt_tag: str, the data tag to use in the hdf5 file for the prostate ground truth
            loaded_dataset: dict, dictionary in which we save the data loaded from the .hdf5 in order to load it much
             faster the next time we need it. If loaded_dataset is None then it doesn't save the data in memory.
        """
        self.key_list = key_list
        self.batch_size = batch_size
        self.file_path = file_path
        self.input_size = input_size
        self.channels = channels
        self.preproc_fn = preproc_fn
        self.final_gt_tag = final_gt_tag
        self.prostate_gt_tag = prostate_gt_tag
        self.loaded_dataset = loaded_dataset

    def __len__(self):
        return int(np.ceil(len(self.key_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        """This method loads and returns slices from one batch.

        Args:
            idx: int, index of the sequence

        Returns:
            x: array, array of slices
            y: list of array, array of ground truth (for the prostate and final segmentation)
        """
        keys = self.key_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = []
        y_full = []
        y_prostate = []

        with h5py.File(self.file_path, 'r') as f:
            for key, i in keys:
                if not self.loaded_dataset or key not in self.loaded_dataset.keys():
                    # If there is only one ground truth, gt_prostate is None
                    img, gt, gt_prostate = get_data(key, f, self.channels, self.final_gt_tag)
                    img = preprocess_channel(img, self.preproc_fn)

                    if self.loaded_dataset is not None:
                        self.loaded_dataset[key] = {"img": img, "gt": gt, "gt_prostate": gt_prostate}
                else:
                    img, gt, gt_prostate = self.loaded_dataset[key].values()

                img_slice = img[int(i)]
                gt_slice = gt[int(i)]

                img_slice = centered_resize(img_slice, (self.input_size, self.input_size))
                gt_slice = centered_resize(gt_slice, (self.input_size, self.input_size))

                x.append(img_slice)
                y_full.append(gt_slice)

                gt_prostate_slice = gt_prostate[int(i)]
                gt_prostate_slice = centered_resize(gt_prostate_slice, (self.input_size, self.input_size))
                y_prostate.append(gt_prostate_slice)

        y = [np.array(y_prostate), np.array(y_full)]

        return np.array(x), y


class TestDataSequence(Sequence):
    """ Sequence of images for Keras.
        This sequence does not use a shuffled list, instead this sequence returns one patient at a time for
        predicting and saving results.
    """

    def __init__(self, set, file_path, input_size, channels, preproc_fn=(), subfold_index=None):
        """
        Args:
            set: string, which set to load ('train', 'validation', 'test')
            file_path: string, path to the hdf5 dataset file
            input_size: int, size of the image (for both width and height) in pixels
            channels: list of int, input channels to use. 0 includes T2-w, 1 includes ADC. Both modality can be used
                at the same time (2 input channels).
            preproc_fn: tuple, preprocessing functions to be applied on the data
            subfold_index: int, the validation subfold index in case of a cross-validation experiment

        """
        self.file_path = file_path
        self.input_size = input_size
        self.channels = channels
        self.preproc_fn = preproc_fn
        self.set = set
        self.subfold_index = subfold_index

        with h5py.File(self.file_path, 'r') as f:
            # In case of a cross-validation experiment, load the appropriate subfold's patients
            if self.subfold_index is not None:
                subfold_keys = load_subfolds_in_set(set_name=self.set, file=f, subfold_index=self.subfold_index)
                # Path to the patient keys are the concatenation of the subfold group and patient group
                self.key_list = flatten(
                    [[pjoin(subfold_key, patient_key) for patient_key in f[subfold_key].keys()]
                     for subfold_key in subfold_keys]
                )

            else:
                self.key_list = [pjoin(self.set, set_key) for set_key in f[self.set].keys()]

    def __len__(self):
        return int(np.floor(len(self.key_list)))

    def __getitem__(self, idx):
        """This method loads are returns slices from one patient.

        Args:
            idx: int, Index of the sequence

        Returns:
            img_name: string, name of the patient in the dataset
            x: array, array of slices
            y: list of array, array of ground truth (for the prostate and final segmentation)
            voxel_size: list, size of the voxels
        """
        img_name = self.key_list[idx]

        with h5py.File(self.file_path, 'r') as f:
            img, gt, gt_prostate = get_data(img_name, f, self.channels)
            voxel_size = f[img_name].attrs.get('voxel_size', default=[1, 1, 1])

        img = preprocess_channel(img, self.preproc_fn)

        x = []
        y_full = []
        y_prostate = []

        for i in range(img.shape[0]):
            x.append(centered_resize(img[i], (self.input_size, self.input_size)))
            y_full.append(centered_resize(gt[i], (self.input_size, self.input_size)))
            y_prostate.append(centered_resize(gt_prostate[i], (self.input_size, self.input_size)))

        y = [np.array(y_prostate), np.array(y_full)]

        return basename(img_name), np.array(x), y, voxel_size


def get_data(key, file, channels, final_gt_tag="gt", prostate_gt_tag="gt_prostate"):
    """ Get data corresponding to the given patient key : the mri volumes and associated ground truth.

    Args:
        key: str, path to a patient in the hdf5 file in a "<set>/<patient_id>" format
        file: h5pyFile object, the hdf5 containing CLARA-P data
        channels: list of int, input channels to use. 0 includes T2-w, 1 includes ADC. Both modality can be used at the
         same time (2 input channels).
        final_gt_tag: str, the data tag to use in the hdf5 file for the final ground truth
        prostate_gt_tag: str, the data tag to use in the hdf5 file for the prostate ground truth

    Returns:
        a tuple of:
            img: numpy array, the patient mri volume of shape (num_slices, width, height, num_channels)
            gt: numpy array, the corresponding ground truth mask of shape (num_slices, width, height, num_classes)
            gt_prostate: numpy array, the corresponding prostate ground truth mask of shape (num_slices, width, height,
             num_classes=2).
    """
    img = np.array(file[f'{key}/img'])[:, :, :, channels]
    gt = np.array(file[f'{key}/{final_gt_tag}'])
    gt_prostate = np.array(file[f'{key}/{prostate_gt_tag}'])

    return img, gt, gt_prostate


def preprocess_channel(img, preproc_fn):
    """ This method preprocesses an image by channel.

    Args:
        img: hdf5 data, img in 3d like this ('s', 0, 1, 'c'):
                - 's': number of slices.
                - 0, 1: x and y dim.
                - 'c': number of channels.
        preproc_fn: tuple of functions used for preprocessing

    Returns:
        ndarray of the image with preprocesses applied on each channel.
    """

    # Convert the image in float32 (GPU limitation)
    img = np.array(img).astype(np.float32)

    # Preprocessing over each channel
    for c in range(img.shape[-1]):
        for pre in preproc_fn:
            img[..., c] = pre(img[..., c])

    return img


def load_subfolds_in_set(set_name, file, subfold_index):
    """ In case of a cross-validation dataset, returns the subfold groups associated to the specified set.

    Args:
        set_name: str, name of the concerned set ('train' or 'validation')
        file: HDF5 object, the cross-validation dataset file
        subfold_index: int, the index of the validation subfold for the experiment

    Returns:
        a list of str, contains the name of the subfold groups composing the set
    """
    validation_key = f'subfold_{subfold_index}'

    if set_name == Dataset.VAL_KEY:
        return [validation_key]

    elif set_name == Dataset.TRAIN_KEY:
        train_keys = list(file.keys())
        train_keys.remove(validation_key)
        return train_keys


if __name__ == '__main__':
    """ This main is only used for code testing. """

    from matplotlib import pyplot as plt
    import argparse

    aparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    aparser.add_argument("path", help="Path to the hdf5 dataset", type=str)
    aparser.add_argument("--classes", help="Number of classes", type=int, default=6)
    aparser.add_argument("--crop", help="Size of the input images after cropping", type=int, default=96)
    aparser.add_argument("--channels", type=int, nargs='+', default=[0, 1],
                         help="Channels to use in the input. 0 includes T2-w, 1 includes ADC. "
                              "Both are used by default.")
    aparser.add_argument("--subfold", help="The index of the validation subfold for this experiment", type=int)

    args = aparser.parse_args()

    ds = Dataset(path=args.path, nb_classes=args.classes, channels=args.channels, crop_size=args.crop, batch_size=1,
                 subfold_index=args.subfold)

    ds.view_sample(channels=args.channels)
