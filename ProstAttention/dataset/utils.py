import logging
import os
from os.path import join as pjoin

import numpy as np
from keras.utils import to_categorical


def get_image_path(patient_dir, patient_id, include_pattern, exclude_pattern=None):
    """ Get the path to an image of the specified modality by screening the path to the patient directory.

    Args:
        patient_dir: str, path to the patient directory
        patient_id: str, patient identifier
        include_pattern: str, pattern for the required image (e.g. "T2", "ADC", "ADC-computed", "GT"). Filenames
         starting with this pattern will be searched.
        exclude_pattern: str, pattern corresponding to images to exclude (e.g. "ADC-computed"). Filenames starting with
         this pattern will be removed from the results path.

    Returns:
        a str, the path to the patient ground truth nifti file. If there is not exactly 1 ground truth file, returns
         None.
    """
    assert include_pattern != exclude_pattern, "Image inclusion pattern must be different from the exclusion pattern."
    image_path = None
    filenames = [filename for filename in os.listdir(patient_dir) if filename.startswith(include_pattern)]
    if exclude_pattern is not None:
        filenames = [filename for filename in filenames if not filename.startswith(exclude_pattern)]
    if len(filenames) == 1:
        image_path = pjoin(patient_dir, filenames[0])
    elif len(filenames) > 1:
        logging.warning(f"patient {patient_id}: several {include_pattern} files available {filenames}")
    else:
        logging.warning(f"patient {patient_id}: no {include_pattern} images available")

    return image_path


def format_groundtruth(groundtruth, nb_classes=2):
    """ Converts groundtruths from (height, width, nb_slices) format to (nb_slices, height, width, nb_classes) one-hot
    vector format.

    Args:
        groundtruth: numpy array of shape (height, width, nb_slices), contains ground truth label for each pixel
        nb_classes: int, number of classes in data (2 for prostate mask, nb_classes for the final mask)

    Returns:
        numpy array of shape (nb_slices, height, width, nb_classes), groundtruth mask in a categorical format.
    """
    gt = groundtruth.transpose(2, 0, 1)
    # Note : to_categorical also constitutes a check on label values : returns an IndexError if a label is
    # not in range(0, nb_classes)
    return to_categorical(gt, num_classes=nb_classes).astype(np.uint8)


def apply_argmax_to_set(set_array):
    """ Apply argmax to a set of predictions or ground truths with categorical encoding.

    Args:
        set_array: array of shape (nb_patients, nb_slices, W, H, nb_classes) if all the patients have the same
         nb_slices, otherwise shape is (nb_patients,)

    Returns:
         array of shape (nb_patients, nb_slices, W, H).
    """
    # Prediction shape is one when patient do not have the same number of slices
    if len(set_array.shape) == 1:
        return np.concatenate([apatient.argmax(axis=-1).flatten() for apatient in set_array])
    else:
        return set_array.argmax(axis=-1).flatten()
