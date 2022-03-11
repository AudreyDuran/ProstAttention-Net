import keras.backend as K


def reduce_crossentropy(crossentropy):
    """ Compute mean of crossentropy.

    Args:
        crossentropy: keras_var, as returned by  keras.losses.categorical_crossentropy

    Returns:
        the mean crossentropy value
    """
    num_elem = K.tf.cast(K.tf.size(crossentropy), dtype=K.tf.float32)
    return K.tf.reduce_sum(crossentropy) / (num_elem + K.epsilon())
