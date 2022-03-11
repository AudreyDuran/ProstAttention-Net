import os
import warnings
from os.path import join as pjoin

import h5py
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm

from ProstAttention.dataset.image_processing.ANTsProcessing import ANTsProcessing
from ProstAttention.dataset.utils import (
    get_image_path,
    format_groundtruth
)

from ProstAttention.project_utils.utils import init_logging


def initialize_hdf5_set_groups(h5py_file, patient_to_group_dict):
    """ Create groups for the different sets ('train', 'validation' and 'test' or 'subfold_<i>' in case of
    cross-validation ) in the hdf5 file.

    Args:
        h5py_file: h5pyFile object, the hdf5 data will be written to
        patient_to_group_dict: dictionary, contains group ('train', 'val' or 'subfold_0', 'subfold_1' etc) for each
         patients.
    """
    for subfold_name in set(patient_to_group_dict.values()):
        h5py_file.create_group(subfold_name)


def assign_set_group_to_patients(path_to_db, set_split, subfold_file):
    """ Assigns a group ('train', 'validation', 'test' or 'subfold_1', 'subfold_2' etc. in case of cross-validation) for
     all patients in the database. A random split is computed given set_split proportion for a train/val/test
     experiment, while the split specified is subfold_file is used in case of cross-validation experiment.

    Args:
        path_to_db: str, path to the Nifti dataset folder
        set_split: list of float, contains 3 values for each set proportion
        subfold_file: str, path to the csv file containing subfold composition.

    Returns:
        a dictionary, keys are patients id and values the corresponding set group
    """
    # One folder per patient
    patients_list = [name for name in os.listdir(path_to_db) if os.path.isdir(pjoin(path_to_db, name))]

    if set_split:
        assert abs(1 - sum(set_split)) < 1e-6, 'Train/validation/test split must sum to 1'

        prop_valid = set_split[1]
        prop_test = set_split[2]
        num_valid = int(len(patients_list) * prop_valid)
        num_test = int(len(patients_list) * prop_test)

        shuffled_indices = np.random.permutation(len(patients_list))
        indices_valid = shuffled_indices[:num_valid]
        indices_test = shuffled_indices[num_valid:num_valid + num_test]
        indices_train = shuffled_indices[num_valid + num_test:]

        # Convert to numpy array to use list indexing
        patients_list = np.array(patients_list)

        patient_to_group = {patient_train: 'train' for patient_train in patients_list[indices_train]}
        patient_to_group.update({patient_valid: 'validation' for patient_valid in patients_list[indices_valid]})
        patient_to_group.update({patient_test: 'test' for patient_test in patients_list[indices_test]})
    else:
        subfold_split_df = pd.read_csv(f'{subfold_file}', dtype={'patient': str, 'subfold': str})
        patient_to_group = subfold_split_df.set_index('patient')['subfold'].to_dict()
        assert len(patients_list) == len(list(patient_to_group.keys())), \
            f"The number of patient's folders ({len(patients_list)}) is different from the number of patients in the" \
            f" subfold split file ({len(list(patient_to_group.keys()))})."
    return patient_to_group


def create_patient_group(h5py_file, patient_to_group_dict, patient_dir, additional=None):
    """ Create group (either in the classic groups 'train', 'validation', 'test' or 'subfold_0', 'subfold_1' etc in case
    of a cross-validation experiment) for the specified patient.

    Args:
        h5py_file: h5pyFile object, the hdf5 data will be written to
        patient_to_group_dict: dictionary, contains group ('train', 'validation', 'train' or 'subfold_<i>') for each
         patients
        patient_dir: str, patient id
        additional: str, additional information to write in the patient groupe name

    Returns:
        h5pyGroup object, the created patient group in the hdf5 file
    """
    g = patient_to_group_dict[patient_dir]
    group = h5py_file.get(g)
    patient_group = f'{patient_dir}'
    if additional:
        patient_group = patient_group + f'{additional}'
    return group.create_group(patient_group)


def save_patient_data_and_attributes(patient, nb_classes, t2_img, adc_img, gt, gt_prostate, voxel_size, origin):
    """ Format and save patient images data and attributes to hdf5.

    Args:
        patient: h5pyGroup object, the patient group in the hdf5 file
        nb_classes: int, number of classes in the dataset
        t2_img: numpy array, contains T2 data
        adc_img: numpy array, contains ADC data
        gt: numpy array, contains the groundtruth label
        gt_prostate: numpy array, contains the contour of the prostate
        voxel_size: tuple, the voxel sizes in millimeters
        origin: tuple, the image's origin
    """
    gt = format_groundtruth(groundtruth=gt, nb_classes=nb_classes)
    gt_prostate = format_groundtruth(groundtruth=gt_prostate, nb_classes=2)

    # The final image shape is (nb_slices, height, width, nb_modalities=2)
    img = np.stack((t2_img, adc_img), axis=-1)
    img = img.transpose((2, 0, 1, 3))

    patient.create_dataset('img', data=img)
    patient.create_dataset('gt', data=gt)
    patient.create_dataset('gt_prostate', data=gt_prostate)

    patient.attrs['voxel_size'] = voxel_size
    patient.attrs['origin'] = origin


def generate_dataset(path, hdf5_filename, nb_classes, target_pixel=None, set_split=[0.6, 0.2, 0.2], subfold_file=None):
    """
    Create a hdf5 file for all images in path directory or in subfolds_patient_to_group in the case of a cross
    validation experiment.

    Args:
        path: str, path to the Nifti dataset folder
        hdf5_filename: str, filename of the hdf5 file to be created, or filename prefix if using subfolds (cross-valid
         experiment)
        nb_classes: int, number of classes in the dataset
        target_pixel: float, the target pixel size for both width and height, in mm
        set_split: list containing 3 values, train split
        subfold_file: str, path to the csv file containing subfold composition. None if the experiment is not a cross-
        validation but a train/val/test split.
    """
    # Fix the seed so that the 3 sets always contain the same patients
    np.random.seed(7)

    patient_to_group = assign_set_group_to_patients(path_to_db=path, set_split=set_split, subfold_file=subfold_file)

    image_processing = ANTsProcessing(target_pixel=target_pixel)

    errors = {}

    with h5py.File(hdf5_filename, 'a') as h5f:
        # Note: comment the line below to complete an existing hdf5 with additional data
        initialize_hdf5_set_groups(h5py_file=h5f, patient_to_group_dict=patient_to_group)

        for patient_dir in tqdm(patient_to_group.keys(), desc="Generating hdf5 : "):
            dir_path = pjoin(path, patient_dir)
            # MRI T2 data
            t2_img_path = get_image_path(patient_dir=dir_path, patient_id=patient_dir, include_pattern="T2")
            if t2_img_path is None:
                errors[patient_dir] = 'T2 not found, patient excluded from hdf5 file.'
                continue

            t2_img_object = image_processing.load_nifti_image(nifti_image_path=t2_img_path)

            # Check that the volume has the expected dimension
            if image_processing.get_image_dimension(t2_img_object) != 3:
                errors[patient_dir] = f'shape T2 {t2_img_object.shape}'
                continue

            patient = create_patient_group(
                h5py_file=h5f,
                patient_to_group_dict=patient_to_group,
                patient_dir=patient_dir
            )

            t2_img_object_resampled = image_processing.resample_image(image_object=t2_img_object)
            t2_img = image_processing.convert_image_to_numpy_array(image_object=t2_img_object_resampled,
                                                                   dtype='float32')

            # Add MRI ADC data if exists
            adc_img_path = get_image_path(patient_dir=dir_path,
                                          patient_id=patient_dir,
                                          include_pattern="ADC")
            if adc_img_path is not None:
                adc_img_object = image_processing.load_nifti_image(nifti_image_path=adc_img_path)

                adc_img_resampled = image_processing.resample_image_to_target_image(
                    image_object=adc_img_object,
                    target_image_object=t2_img_object_resampled
                )

                adc_img = image_processing.convert_image_to_numpy_array(image_object=adc_img_resampled, dtype='float32')

            else:
                adc_img = np.zeros_like(t2_img)

            # GT data
            gt_mask = np.zeros_like(t2_img)  # creates an image that may be overwritten
            gt_prostate = np.zeros_like(t2_img)

            gt_path = get_image_path(patient_dir=dir_path,
                                     patient_id=patient_dir,
                                     include_pattern="GT")

            if gt_path is not None:
                ni_gt = image_processing.load_nifti_image(nifti_image_path=gt_path)
                gt_mask_resampled = image_processing.resample_groundtruth_mask(
                    groundtruth_mask=ni_gt,
                    target_image_object=t2_img_object_resampled
                )
                gt_mask = image_processing.convert_image_to_numpy_array(image_object=gt_mask_resampled)

                # Compute prostate segmentation mask
                gt_prostate = np.clip(gt_mask, 0, 1)

            patient_attributes = {
                'voxel_size': image_processing.get_voxel_size(image_object=t2_img_object_resampled),
                'origin': image_processing.get_origin(image_object=t2_img_object_resampled)
            }

            patient_imgs = {
                't2_img': t2_img,
                'adc_img': adc_img,
                'gt': gt_mask,
                'gt_prostate': gt_prostate
            }
            save_patient_data_and_attributes(patient=patient, nb_classes=nb_classes, **patient_imgs,
                                             **patient_attributes)

    if errors:
        print(pd.DataFrame.from_dict(errors, orient='index', columns=['ERRORS']))


if __name__ == '__main__':
    init_logging()
    import argparse

    aparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    aparser.add_argument("path", help="Path to the DatasetFolder containing the nifti files for each patient (MRIs and "
                                      "groundtruths)", type=str)
    aparser.add_argument("name", help="Name of the hdf5 file to generate", type=str)
    aparser.add_argument("--classes", type=int, default=6, help="Number of classes in the dataset.")
    aparser.add_argument("--subfold", type=str, help="Path to the csv file containing subfold composition. To use in "
                                                     "case of a cross-validation experiment.")
    aparser.add_argument("--set_split", type=float, nargs='+', help="Proportion to use for the patient split into \
        train/validation/test sets (ex: [0.6, 0.2, 0.2]). For a cross-validation experiment, the subfold file \
        repartition is used.")
    aparser.add_argument("--target_pixel", type=float, help="The target pixel size for both width and height, in mm. "
                                                            "The resolution in z is kept unchanged", default=1.)

    args = aparser.parse_args()

    if (args.subfold and args.set_split) or (not args.subfold and not args.set_split):
        raise aparser.error("Both or none subfold and set_split arguments were specified. Specify subfold only for a "
                            "cross-validation experiment, or set_split for a train/val/test experiment with random "
                            "patient split.")

    generate_dataset(path=args.path, hdf5_filename=args.name, nb_classes=args.classes, target_pixel=args.target_pixel,
                     set_split=args.set_split, subfold_file=args.subfold)
