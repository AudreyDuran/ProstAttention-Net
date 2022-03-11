import ants


class ANTsProcessing:
    def __init__(self, target_pixel=None, registration=False):
        """
        Args:
            target_pixel: float, the target pixel size for both width and height, in mm
            registration: bool, whether to register the image on the target image or not
        """
        self.target_pixel = target_pixel
        self.registration = registration

    @staticmethod
    def load_nifti_image(nifti_image_path):
        """ Load a nifti image from its path.

        Args:
            nifti_image_path: str, path to the nifti image

        Returns:
             an ANTsImage object
        """
        return ants.image_read(nifti_image_path)

    @staticmethod
    def convert_image_to_numpy_array(image_object, dtype='uint16'):
        """ Returns image data in a numpy array.

        Args:
            image_object: ANTsImage object, the image object
            dtype: str, data type of the numpy array to return

        Returns:
            a numpy array, contains image data
        """
        return image_object.numpy().astype(dtype)

    def resample_image(self, image_object, interpolation=0):
        """ Resample an image to the given pixel size using ANTs library.

        Args:
            image_object: ANTsImage object, the image object to resample
            interpolation: int, the interpolation type to use (0 for linear and 1 for nearest neighbor)

        Returns:
            an ANTsImage object, contains the image resampled to the target pixel size.
        """
        return ants.resample_image(
            image=image_object,
            resample_params=(self.target_pixel, self.target_pixel, image_object.spacing[2]),
            interp_type=interpolation
        )

    def resample_image_to_target_image(self, image_object, target_image_object, interpolation='linear',
                                       is_masked_image=False):
        """ Resample an image by using another image as target reference (e.g. ADC image to T2 image).

        Args:
            image_object: ANTsImage object, the image to resample
            target_image_object: ANTsImage object, the reference image
            interpolation: str, the interpolation type to use ('linear', 'nearestNeighbor' or 'genericLabel')
            is_masked_image: bool, whether the image to resample to the target is a label mask or not

        Returns:
            an ANTsImage object, the image resampled to the target image
        """
        resampled_image = ants.resample_image_to_target(
            image=image_object,
            target=target_image_object,
            interp_type=interpolation
        )
        if self.registration and not is_masked_image:
            resampled_image = self.register_to_image(
                image_object=resampled_image,
                target_image_object=target_image_object
            )

        return resampled_image

    @staticmethod
    def register_to_image(image_object, target_image_object):
        """ Register an image to a target image (usually ADC on T2 target).

        Args:
            image_object: ANTsImage object, the image to register (moving image)
            target_image_object: ANTsImage object, the reference image on which to register the moving image

        Returns:
            an ANTsImage object, the registered image on the reference
        """
        d_registration = ants.registration(
            fixed=target_image_object,
            moving=image_object,
            type_of_transform='Rigid'
        )
        return ants.apply_transforms(
            fixed=target_image_object,
            moving=image_object,
            transformlist=d_registration['fwdtransforms']
        )

    def resample_groundtruth_mask(self, groundtruth_mask, target_image_object):
        """ Resample groundtruth mask alike target images, using ants.resample_image_to_target method. Note that the
        absolutely same image is obtained with ants.resample_imag with interp_type=1 (nearest neighbor) as with
        ants.resample_image_to_target method with interp_type='nearestNeighbor'. Here, ants.resample_image_to_target is
        the chosen method because of the available interpolation type 'genericLabel' (between 60 and 100 voxels per
        volume differences i.e. 0.006 % of the volume of differences).

        Args:
            groundtruth_mask: ANTsImage object, the groundtruth mask to resample
            target_image_object: ANTsImage object, the reference image

        Returns:
            an ANTsImage object, the groundtruth mask resampled to the target image
        """
        return self.resample_image_to_target_image(image_object=groundtruth_mask,
                                                   target_image_object=target_image_object,
                                                   interpolation='genericLabel',
                                                   is_masked_image=True)

    @staticmethod
    def get_image_dimension(image_object):
        """ Get the image dimension.

        Args:
            image_object: the API's image object

        Return:
            int, the number of dimension in the image
        """
        return len(image_object.shape)

    @staticmethod
    def get_voxel_size(image_object):
        """ Get the voxel sizes in millimeters of an image object.

        Args:
            image_object: ANTsImage object, the image to get the voxel size of

        Returns:
            tuple, the voxel sizes in millimeters
        """
        return image_object.spacing

    @staticmethod
    def get_origin(image_object):
        """ Get the origin of an image object.

        Args:
            image_object: a ANTsImage object, the image to get the origin of

        Returns:
            tuple, the image's origin
        """
        return image_object.origin

    @staticmethod
    def save_image(image_object, output_filename):
        """ Save the given image object to file.

        Args:
            image_object: ANTsImage object, the image to save to file
            output_filename: str, the path to the output filename
        """
        ants.image_write(image_object, output_filename)
