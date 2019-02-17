"""
A Keras implementation of Grad-CAM from Selvaraju et al "Grad-CAM: Visual
Explanations from Deep Networks via Gradient-based Localization" (2017).

Original paper @ arXiv:1610.02391v3

Code based in large part on public github implementation:
https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py
First retrieved 29-NOV-2018
"""

import numpy as np
import tensorflow as tf

import keras.backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.convolutional import _Conv


def grad_cam(input_model, image, category_index=None, layer_name=None, output_layer_name=None):
    """
    Calculates the class activation map according to the Grad-CAM algorithm
    and returns the result of the calculation as a heatmap to be superimposed
    on the input.

    TODO: determine a good place to incorporate this function

    Paramters:
    ---------------------------------------------------------------------------
    input_model : Keras model obj
        A Keras model, eg, VGG16, ResNet50, etc. Must be trained already!

    image : np.array
        The image is assumed to have been passed through a preprocessor
        suitable for the input_model object prior to calling grad_cam.

    category_index : int, default None
        If supplied, this is the category index for which evidence will
        be calculated. If not supplied, the most probable class (as
        determined by the model predictions) will be used instead.

    layer_name: str, default None
        If supplied, evidence will be taken from this layer. If not
        supplied, this function tries to identify the final convolutional
        layer in the model.

    output_layer_name : str, default None
        The name of the output layer in the model. If not supplied, it
        is assumed that the output layer is the last layer in
        input_model.layers.

    Returns:
    ---------------------------------------------------------------------------
    CAM : np.array
        The class activation map

    """

    # =========================================================================
    #  Extract information from input_model
    #  We need the following information from the model:
    #   1. The reference to the final convolutional layer
    #   2. The number of classes
    #   3. If not given, the class for which to find evidence
    # =========================================================================

    # TODO: make the input layer something that can be specified.
    # this will probably be important for models that concatenate
    # learned deep features with structured data
    input_ = input_model.layers[0].input

    # find either the user-specified output or the output of the
    # final convolutional layer
    if layer_name:
        conv_output = _get_layer_by_name(input_model, layer_name)
    else:
        for layer in input_model.layers[::-1]:
            if isinstance(layer, _Conv):
                conv_output = layer.output
                break

    # infer the number of classes from the model. if the name of the output
    # layer is not supplied (eg, VGG16 uses "predictions" as name of final
    # layer, while ResNet50 uses 'fc1000'), then we take the last layer
    if output_layer_name:
        output_ = _get_layer_by_name(input_model, output_layer_name)
    else:
        output_ = input_model.layers[-1]
    nclasses = output_.output_shape[-1]

    # if we dont have a category_index, we run the image through the model
    # and find the most likely class. however, keeping this as a kwarg allows
    # us to look at the evidence for other classes
    if not category_index:
        category_index = input_model.predict(image).argmax()
    if category_index >= nclasses:
        msg = 'category index %s exceeds number of classes %s'
        raise ValueError(msg % (category_index, nclasses))

    # =========================================================================
    #   Modify model to support GradCAM calculation
    #   Here, we add a final one-hot layer to the model to determine if we
    #   predicted the target category or not
    # =========================================================================

    target_layer = lambda x: _hit(x, category_index, nclasses)
    last = Lambda(target_layer, output_shape=_hit_shape)(output_.output)
    model = Model(inputs=input_, outputs=last)

    # =========================================================================
    #   Calcuation of class activation map (Selvaraju Eq1, Eq2)
    # =========================================================================

    # Selvaraju Eq 1:
    # a_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A_{i,j}^k}
    # The normalization appears to be additional compared to paper, but
    # seems to exist just to ensure that our images stay sensible
    loss = K.sum(model.layers[-1].output)
    grads = _normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    # output = A_k
    # grads_val = \frac{\partial y^c}{\partial A_{i,j}^k}
    # weights is the global average pooling of grads_val (average over all space)
    output, grads_val = gradient_function([image])
    output, grads_val = output.squeeze(), grads_val.squeeze()
    weights = np.mean(grads_val, axis=tuple(range(grads_val.ndim - 1)))

    # Selvaraju Eq 2 (inner)
    # this is the canonical method of CAM formation: sumproduct
    # of weights and final-convolution feature activations
    cam = np.ones(output.shape[:output.ndim - 1], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[..., i]

    # Selvaraju Eq 2 (outer/RELU)
    # cam = np.maximum(cam, 0)

    heatmap = cam / np.max(np.abs(cam))
    return heatmap


# =============================================================================
#   helper functions
# =============================================================================

def _hit(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def _hit_shape(input_shape):
    return input_shape


def _normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def _get_layer_by_name(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer


def _compute_gradients(tensor, vars_):
    grads = tf.gradients(tensor, vars_)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(vars_, grads)]