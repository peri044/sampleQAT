import tensorflow as tf
import quantize_wrapper

def quantize_layer(layer):
    """
    Quantize a convolution, Dense layer and MaxPool Layer.
    By default, we quantize weights per channel and the inputs per tensor.
    This function identifies a conv or FC layer and wraps it with QDQ nodes.
    """
    layer_wrapper = layer
    if isinstance(layer, tf.keras.layers.Conv2D) or \
       isinstance(layer, tf.keras.layers.DepthwiseConv2D) or \
       isinstance(layer, tf.keras.layers.Dense):

        layer_wrapper = quantize_wrapper.QuantizeWrapper(layer, quantize_inputs=True, quantize_weights=True)
    # elif isinstance(layer, tf.keras.layers.MaxPool2D):
        # layer_wrapper = quantize_wrapper.QuantizeWrapper(layer, quantize_inputs=True)

    return layer_wrapper

def quantize_model(model):
    """
    clone the model and apply quantization at required layers
    """
    model = tf.keras.models.clone_model(model, \
                    input_tensors=None, \
                    clone_function=quantize_layer)

    return model
