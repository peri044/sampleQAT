  # Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper which applies quantization operations over underlying layer.

   `QuantizeWrapper` is responsible for modifying the construction of the
   underlying layer to ensure proper quantization operations are placed in the
   graph.

   These operations ensure proper introduction of inference time losses during
   training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import tf_inspect
import quantizers

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object


class QuantizeWrapper(tf.keras.layers.Wrapper):
  """Quantizes the weights and activations of the keras layer it wraps."""

  def __init__(self, layer, quantize_inputs=False, quantize_weights=False, **kwargs):
    """Create a quantize emulate wrapper for a keras layer.
    This wrapper provides options to quantize inputs or weights of the layer.
    Args:
      layer: The keras layer to be quantized.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """
    if layer is None:
      raise ValueError('`layer` cannot be None.')

    # Check against keras.Model since it is an instance of keras.layers.Layer.
    if not isinstance(layer, tf.keras.layers.Layer) or isinstance(
        layer, tf.keras.Model):
      raise ValueError(
          '`layer` can only be a `tf.keras.layers.Layer` instance. '
          'You passed an instance of type: {input}.'.format(
              input=layer.__class__.__name__))

    self.quantize_inputs = quantize_inputs
    self.quantize_weights = quantize_weights

    if 'name' not in kwargs:
      kwargs['name'] = self._make_layer_name(layer)

    super(QuantizeWrapper, self).__init__(layer, **kwargs)

    self._track_trackable(layer, name='layer')

  @staticmethod
  def _make_layer_name(layer):
    return '{}_{}'.format('quant', layer.name)

  @staticmethod
  def _weight_name(name):
    """Extracts the weight name from the full TensorFlow variable name.

    For example, returns 'kernel' for 'dense_2/kernel:0'.

    Args:
      name: TensorFlow variable name.

    Returns:
      Extracted weight name.
    """
    return name.split(':')[0].split('/')[-1]

  def build(self, input_shape):
    super(QuantizeWrapper, self).build(input_shape)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=tf.keras.initializers.Constant(-1),
        dtype=tf.dtypes.int32,
        trainable=False)

    self._weight_vars = []
    self.input_vars = {}
    # quantize weights only applicable for Conv/FC layers.
    if  self.quantize_weights:
        if isinstance(self.layer, tf.keras.layers.DepthwiseConv2D):
            kernel_weights = getattr(self.layer, 'depthwise_kernel')
        else:
            kernel_weights = getattr(self.layer, 'kernel')

        min_weight = self.layer.add_weight(
            kernel_weights.name.split(':')[0] + '_min',
            shape=(kernel_weights.shape[-1]),
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False)
        max_weight = self.layer.add_weight(
            kernel_weights.name.split(':')[0] + '_max',
            shape=(kernel_weights.shape[-1]),
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False)
        quantizer_vars = {'min_var': min_weight, 'max_var': max_weight}
        self._weight_vars.append((kernel_weights, quantizer_vars))
          # Needed to ensure unquantized weights get trained as part of the wrapper.
        self._trainable_weights.append(kernel_weights)

    if self.quantize_inputs:
        input_min_weight = self.layer.add_weight(
            self.layer.name + '_min',
            shape=None,
            initializer=tf.keras.initializers.Constant(-6.0),
            trainable=False)
        input_max_weight = self.layer.add_weight(
            self.layer.name + '_max',
            shape=None,
            initializer=tf.keras.initializers.Constant(6.0),
            trainable=False)
        self.input_vars['min_var'] = input_min_weight
        self.input_vars['max_var'] = input_max_weight

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def _make_quantizer_fn(self, x, training, quantizer_vars, per_channel=False):
    """Use currying to return True/False specialized fns to the cond."""

    def quantizer_fn():
      return quantizers.LastValueQuantize(x, quantizer_vars['min_var'], \
                                          quantizer_vars['max_var'], \
                                          per_channel=per_channel,
                                          is_training=training,
                                          num_bits=8,
                                          narrow_range=True,
                                          symmetric=True)

    return quantizer_fn

  def _mvg_avg_quantizer(self, x, training, quantizer_vars, per_channel=False):
    """Use currying to return True/False specialized fns to the cond."""

    return quantizers.MovingAvgQuantize(x, quantizer_vars['min_var'], \
                                        quantizer_vars['max_var'], \
                                        per_channel=per_channel,
                                        is_training=training,
                                        num_bits=8,
                                        narrow_range=True,
                                        symmetric=True)

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    # Quantize all weights, and replace them in the underlying layer.
    if self.quantize_weights:
        quantized_weights = []
        quantized_weight = self._make_quantizer_fn(self._weight_vars[0][0], training,
                                      self._weight_vars[0][1], per_channel=True)
        quantized_weights.append(quantized_weight())
        # Replace the original weights with QDQ weights
        if isinstance(self.layer, tf.keras.layers.DepthwiseConv2D):
            setattr(self.layer, 'depthwise_kernel', quantized_weights[0])
        else:
            setattr(self.layer, 'kernel', quantized_weights[0])

    # Quantize inputs to the conv layer
    if self.quantize_inputs:
        quantized_inputs = self._mvg_avg_quantizer(inputs, training,
                                      self.input_vars, per_channel=False)
    else:
        quantized_inputs = inputs

    args = tf_inspect.getfullargspec(self.layer.call).args
    if 'training' in args:
      outputs = self.layer.call(quantized_inputs, training=training)
    else:
      outputs = self.layer.call(quantized_inputs)

    return outputs

  def get_config(self):
    base_config = super(QuantizeWrapper, self).get_config()
    config = {'quantize_config': None}
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    # QuantizeWrapper may be constructed with any QuantizeConfig and the
    # wrapper itself cannot know all the possible config classes.
    # The deserialization code should ensure the QuantizeConfig is in keras
    # serialization scope.
    quantize_config = deserialize_keras_object(
        config.pop('quantize_config'),
        module_objects=globals(),
        custom_objects=None)

    layer = tf.keras.layers.deserialize(config.pop('layer'))

    return cls(layer=layer, quantize_config=quantize_config, **config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses
