import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras.backend as K
from VITALabAI.model.semanticsegmentation.unet import VITALUnet
from keras import Input, Model
from keras.layers import MaxPooling2D, Conv2D, Lambda
from keras.optimizers import Adam

from ProstAttention.base_segmentation_model import BaseSegmentationModel
from ProstAttention.dataset.generator import Dataset

from ProstAttention.project_utils.utils import init_logging


class ProstAttentionNet(BaseSegmentationModel):
    """ ProstAttentionNet model.

    Attributes:
        layer_attention: bool, whether to use prostate prediction as an attention mask at each upsampling layer or
             only at the network's bottleneck.
        down_factor: int, the down-sampling factor between the input image and the encoded image.
        **kwargs: additional parameters for the generic pipeline (in BaseSegmentationModel).
    """

    down_factor = 16

    def __init__(self, layer_attention, **kwargs):
        """ Inits ProstAttentionNet.
        """
        self.layer_attention = layer_attention
        super().__init__(**kwargs)

    def encoder(self, input_image):
        """ Encode input image in the latent space. Output shape is
         (input_size/self.down_factor, input_size/self.down_factor, input_shape / (input_size/self.down_factor)**2).

        Args:
            input_image: tensor, input image to downsample.

        Returns:
             tuple of:
                encoded_image: tensor, input image downsampled to the point of reaching the bottleneck of the network.
                connected_feature_maps: list, connected feature maps from each downsampling step to concatenate to the
                 result of the upsampling
        """
        encoded_image, connected_feature_maps = VITALUnet.unet_encoder(
            input_image,
            nb_feature_maps=self.nb_feature_maps,
            alpha=self.l2reg
        )
        return encoded_image, connected_feature_maps

    def prostate_decoder(self, encoded_image, arg_from_encoder, num_classes, name, **kwargs):
        """ Decode encoded sample for prostate segmentation (2 classes problem).

        Args:
            encoded_image: tensor, feature maps at the bottleneck of the base model to upsample.
            arg_from_encoder: list, connected feature maps from each downsampling step to concatenate to the result of
             the upsampling
            num_classes: int, number of classes and channels to output.
            name: str, name of the output layer (useful to link loss to a specific output in a multi-output model).

        Returns:
            tensor, the base model prostate segmentation
        """
        return VITALUnet.unet_decoder(
            x=encoded_image,
            num_classes=num_classes,
            nb_feature_maps=self.nb_feature_maps,
            connected_feature_maps=arg_from_encoder,
            alpha=self.l2reg,
            name=name
        )

    def lesion_decoder(self, encoded_image, arg_from_encoder, num_classes, name, **kwargs):
        """ Decode encoded image for lesion segmentation.

         Args:
             encoded_image: tensor, feature maps at the bottleneck of the base model to upsample.
             arg_from_encoder: list, connected feature maps from each downsampling step to concatenate to the result of
              the upsampling
             num_classes: int, number of classes and channels to output.
             name: str, name of the output layer (useful to link loss to a specific output in a multi-output model).
             **kwargs: additional parameters including 'prostate_prediction'

         Returns:
             tensor, the base model lesion segmentation
         """
        if self.layer_attention:
            return unet_att_decoder(
                x=encoded_image,
                prostate_prediction=kwargs['prostate_prediction'],
                num_classes=num_classes,
                nb_feature_maps=self.nb_feature_maps,
                connected_feature_maps=arg_from_encoder,
                alpha=self.l2reg,
                name=name
            )
        else:
            return VITALUnet.unet_decoder(
                x=encoded_image,
                num_classes=num_classes,
                nb_feature_maps=self.nb_feature_maps,
                connected_feature_maps=arg_from_encoder,
                alpha=self.l2reg,
                name=name
            )

    def build_model(self):
        """ This function builds complete Attention models.

        Returns:
            keras.Model object *not* compiled.
        """
        input_image = Input(shape=self.dataset.get_input_shape(), name='input_image')

        encoded_image, arg_from_encoder = self.encoder(input_image)

        # Decode encoded sample for prostate segmentation
        prostate_prediction = self.prostate_decoder(
            encoded_image=encoded_image,
            arg_from_encoder=arg_from_encoder,
            num_classes=2,
            name='prostate_seg'
        )

        # Multiply prostate output with encoded sample representation
        # First down-sample the prostate segmentation to the encoded_sample shape (downsampled by self.down_factor)
        prostate_down = MaxPooling2D(pool_size=(self.down_factor, self.down_factor), strides=None, padding='same')(
            prostate_prediction
        )

        # Multiply each of the feature maps of the encoded_image with prostate_down
        # First create a Lambda layer performing the operation between the encoded sample and prostate_down segmentation
        att_layer = Lambda(attention_product)
        lesion_input = att_layer([prostate_down, encoded_image])

        # Decode lesion_input for lesion segmentation
        lesion_prediction = self.lesion_decoder(
            encoded_image=lesion_input,
            arg_from_encoder=arg_from_encoder,
            num_classes=self.dataset.nb_classes,
            name='lesion_seg',
            prostate_prediction=prostate_prediction
        )

        self.model = Model(
            inputs=[input_image],
            outputs=[prostate_prediction, lesion_prediction]
        )

        print("Attention-model built")
        self.model.summary()

        return self.model

    def losses(self):
        """ Losses used during the training.
        """
        return {
            'prostate_seg': self.model_losses.prostate_combined_loss,
            'lesion_seg': self.model_losses.combined_loss,
        }

    def losses_weights(self):
        """ Weights associated to each branch losses. During the n first epochs (unitl loss_update_epoch), only
        prostate_seg branch is trained.
        Once the prediction is good enough to be used as an attention mask for the lesion_seg branch, lesion_seg weight
        is changed to 1 with LossWeightsCallback. If a pretrained_model is used, both branches are trained from the
        training start.
        """
        return {
            'prostate_seg': 1.,
            'lesion_seg': 0. if self.pretrained_model is None else self.target_loss_weight,
        }

    def metrics(self):
        """ Metrics used during the training.
        """
        return {
            'prostate_seg': [self.model_losses.metrics.prostate_dice],
            'lesion_seg': [self.model_losses.metrics.weighted_classes_dice,
                           self.model_losses.metrics.lesion_dice,
                           self.model_losses.metrics.dice_on_prostate]
        }


def attention_product(tensors):
    """ Function used to define a Lambda layer which multiply a tensor to a list of tensor. In this project, this is
     used to multiply the predicted prostate segmentation to each feature maps of a given layer.

    Args:
        tensors: list of 2 tensors. The first tensor's 2nd channel will be multiplied to each element of the second
         tensor.
         - <tensor_0> has shape (s, h, w, 2)
         - <tensor_1> has shape (s, h, w, c)

    Returns:
        tensor, the result of the feature-maps wise multiplication.
    """
    # Expand tensors[0] dimensions - Equivalent to tensors[0][..., 1][..., np.newaxis]
    prostate_slice = K.slice(tensors[0], (0, 0, 0, 1), (-1, -1, -1, 1))
    return prostate_slice * tensors[1]


def _upsampling_attention(x, prostate_segmentation, factor: int):
    """ Block making the attention part.

    Args:
        x: tensor, feature maps, at the bottleneck of the UNet, to upsample.
        prostate_segmentation: tensor, output U-Net prostate segmentation.

    Returns:
        tensor, the result of the attention multiplication.
    """
    prostate_segmentation = MaxPooling2D(pool_size=(factor, factor), strides=None, padding='same')(
        prostate_segmentation)
    return Lambda(attention_product)([prostate_segmentation, x])


def unet_att_decoder(x, prostate_prediction, num_classes, nb_feature_maps: int, connected_feature_maps,
                     working_axis=-1, alpha=1e-5, name: str = 'final_output'):
    """ Block making up the second branch upsampling part (upsample by a factor 16), of ProstAttention-Net,
    the proposed customized UNet.
    ProstAttention-Net uses the prostate segmentation (computed by a first branch) in a second upsampling branch to
    focus on the prostate gland.

    Args:
        x: tensor, feature maps at the bottleneck of the ProstAttention-Net, to upsample.
        prostate_prediction: tensor, prostate segmentation outputted by the first branch or the network.
        num_classes: int, number of classes and channels to output.
        nb_feature_maps:  int, number of feature maps at beginning of the network, automatic scaling for the
                          following layers.
        connected_feature_maps: list, connected feature maps from each downsampling step to concatenate to the
                                corresponding upsampling steps of the upsampling half.
        working_axis: int, axis on which to apply the batch normalization.
        alpha: float, L2 Regularization coefficient. Default: 1e-5.
        name: name of the output layer (useful to link loss to a specific output in a multi-output model).

    Returns:
        tensor, UNet's segmentation.
    """
    # Skip Connection - fifth layer
    x = VITALUnet.upsampling_block(x, connected_feature_maps[3], nb_feature_maps * 8, working_axis, alpha)
    x = _upsampling_attention(x, prostate_prediction, factor=8)

    # Skip Connection - fourth layer
    x = VITALUnet.upsampling_block(x, connected_feature_maps[2], nb_feature_maps * 4, working_axis, alpha)
    x = _upsampling_attention(x, prostate_prediction, factor=4)

    # Skip Connection - third layer
    x = VITALUnet.upsampling_block(x, connected_feature_maps[1], nb_feature_maps * 2, working_axis, alpha)
    x = _upsampling_attention(x, prostate_prediction, factor=2)

    # Skip Connection - second layer
    x = VITALUnet.upsampling_block(x, connected_feature_maps[0], nb_feature_maps, working_axis, alpha)

    # output layer
    out = Conv2D(num_classes, (1, 1), activation='softmax', name=name)(x)

    return out


if __name__ == '__main__':
    init_logging()

    # Get generic argument parser
    aparser = BaseSegmentationModel.build_generic_parser()

    # Add arguments specific to the model
    aparser.add_argument("--no_layer_attention", dest="layer_attention", action="store_false",
                         help="Do not use prostate prediction as an attention mask at each upsampling layer but only"
                              "in the latent space.")
    aparser.add_argument("--loss_weight", type=float, help="Value that the lesion segmentation loss weight should be"
                                                           " changed to in LossWeightsCallback.", default=1.)
    aparser.add_argument("--loss_update_epoch", type=int, default=20,
                         help="Epoch on which the lesion segmentation loss weight should be changed in"
                              " LossWeightsCallback.")

    args = aparser.parse_args()

    assert len(args.class_weights) == args.classes, f"{len(args.class_weights)} weights were given but the model has " \
                                                    f"{args.classes} classes. Please provide one weight for each " \
                                                    f"class in --class_weights argument."

    ds = Dataset(path=args.path, nb_classes=args.classes, channels=args.channels, crop_size=args.crop,
                 batch_size=args.batch_size, preproc_fn=ProstAttentionNet.get_preprocessing(),
                 subfold_index=args.subfold, load_hdf5=args.load_hdf5)

    optim = Adam(lr=args.lr)

    model_parameters = {'dataset': ds, 'optimizer': optim, 'name': args.name, 'l2reg': args.reg,
                        'nb_feature_maps': args.nb_feature_maps, 'pretrained_model': args.pretrained_model,
                        'layer_attention': args.layer_attention, 'class_weights': args.class_weights,
                        'target_loss_weight': args.loss_weight, 'patience': args.patience, 'nb_epochs': args.nb_epochs}

    if not args.pretrained_model:
        model_parameters['loss_update_epoch'] = args.loss_update_epoch

    model = ProstAttentionNet(**model_parameters)

    print(f"Model created at: {model.name}")

    if args.pretrained_model:
        print("Loading weights...")
        model.load_weights(args.pretrained_model)
        print("Done.")

    if not args.testonly:
        callbacks = model.get_callbacks(
            learning_rate=args.lr,
            early_stopping=args.early_stopping,
            reduce_lr=True,
            update_loss_weights=True if model.pretrained_model is None else False,
            patience=model.patience,
            model_monitor=model.get_model_monitor(),
            target_weight=model.target_loss_weight,
            update_epoch=model.loss_update_epoch
        )
        model.train(epochs=args.nb_epochs, callbacks=callbacks, workers=args.workers)
        model.save()

    model.evaluate()
    model.export_results()
