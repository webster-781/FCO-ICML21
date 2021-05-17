# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a model for EMNIST classification."""

import functools
from abc import abstractmethod
from absl import flags

# import numpy as np
import tensorflow as tf

from optimization.shared import projector_utils
from utils.models import kernelized
from utils.models.augmented_dense import AugmentedDense
from utils import utils_impl

with utils_impl.record_new_flags() as emnist_cr_flags:
  # EMNIST character recognition flags
  flags.DEFINE_enum('emnist_cr_model', 'cnn', ['cnn', '2nn', 'qlr', 'klr', 'lr'],
                    'Which model to use for classification.')
  flags.DEFINE_boolean('emnist_cr_only_digits', False, 'Digits only (default: False)')
  flags.DEFINE_float('emnist_cr_subset_ratio', 0.1, 'subset of dataset')
  flags.DEFINE_integer('emnist_cr_klr_features', 4096, 'number of kernelized features')
  flags.DEFINE_float('emnist_cr_klr_scale', 10.0, 'scale of random kernelized features')


class ProxModel(tf.keras.Model):
  """
  Abstsract base class for Model with Projector
  """
  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.projector(trainable_vars)
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

  @abstractmethod
  def get_config(self):
    pass

  @abstractmethod
  def call(self, inputs, training=None, mask=None):
    pass

## ------ 1. Logistic Regression Model -------
def create_lr_model(only_digits=True):
  """Create a Logistic Linear Model (for federated)

  When `only_digits=True`, the summary of returned model is

  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  reshape (Reshape)          (None, 784)               0         
  _________________________________________________________________
  dense (Dense)              (None, 10)                7850      
  =================================================================
  Total params: 7,850
  Trainable params: 7,850
  Non-trainable params: 0
  _________________________________________________________________

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    use_bias: A boolean that determines whether to use bias for the Logistic linear.

  Returns:
    A `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      AugmentedDense(10 if only_digits else 62,
        name='dense',
        activation=tf.nn.softmax, 
        **(projector_utils.get_lambd1_lambd2())),
  ])
  return model

def create_augmented_lr_model_from_flags(only_digits=True):
  """
  Create AugmentedLRModel with regularizer and projector
  """
  class AugmentedLRModel(ProxModel):
    """
    Logistic Regression with regularizer and projector
    """
    def __init__(self, name=None, only_digits=True, projector=lambda _: None, regularizer=None):
      super(AugmentedLRModel, self).__init__(name=name)
      self.projector = projector
      self.reshaper = tf.keras.layers.Reshape(
          input_shape=(28, 28, 1), target_shape=(28 * 28,))
      self.dense = AugmentedDense(
          10 if only_digits else 62, 
          name ='dense',
          activation='softmax',
          kernel_regularizer=regularizer,
          **(projector_utils.get_lambd1_lambd2()))

    def call(self, inputs):
      return self.dense(self.reshaper(inputs))

  projector, regularizer = \
    projector_utils.build_proj_reg_from_flags()
 
  model = AugmentedLRModel(
      only_digits=only_digits,
      projector=projector,
      regularizer=regularizer)

  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, 28, 28, 1))
  return model

## ------ 2. Kernalized Logistic Regression Model -------
def create_klr_model(kernelized_features=4096, kernelized_scale=10., only_digits=True):
  """Create a Kernelized Logistic Linear Model (for federated)

  When `only_digits=True`, the summary of returned model is

  Model: "sequential"

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    use_bias: A boolean that determines whether to use bias for the Logistic linear.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      kernelized.RandomFourierFeatures(
          output_dim=kernelized_features,
          scale=kernelized_scale,
          kernel_initializer='gaussian'),
      AugmentedDense(10 if only_digits else 62, 
        name='dense',
        activation=tf.nn.softmax,
        **(projector_utils.get_lambd1_lambd2())),
  ])
  return model

def create_augmented_klr_model_from_flags(kernelized_features=4096, kernelized_scale=10.,only_digits=True):
  """
  Create AugmentedKLRModel with regularizer and projector
  """
  class AugmentedKLRModel(ProxModel):
    """
    Kernelized Logistic Regression with regularizer and projector
    """
    def __init__(self, name=None, kernelized_features=4096, kernelized_scale=10., only_digits=True, projector=lambda _: None, regularizer=None):
      super(AugmentedKLRModel, self).__init__(name=name)
      self.projector = projector
      self.flattener = tf.keras.layers.Flatten(
          input_shape=(28, 28, 1))
      self.kernelizer = kernelized.RandomFourierFeatures(
          output_dim=kernelized_features,
          scale=kernelized_scale,
          kernel_initializer='gaussian')
      self.dense = AugmentedDense(
          10 if only_digits else 62, 
          name='dense',
          activation='softmax',
          kernel_regularizer=regularizer,
          **(projector_utils.get_lambd1_lambd2()))

    def call(self, inputs):
      return self.dense(self.kernelizer(self.flattener(inputs)))

  projector, regularizer = \
    projector_utils.build_proj_reg_from_flags()
 
  model = AugmentedKLRModel(
      kernelized_features=kernelized_features, 
      kernelized_scale=kernelized_scale,
      only_digits=only_digits,
      projector=projector,
      regularizer=regularizer)

  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, 28, 28, 1))
  return model


## ------ 3. Quadratic-Feature Map Logistic Regression Model -------
class QuadraticOuter(tf.keras.layers.Layer):
  def __init__(self):
    super(QuadraticOuter, self).__init__()

  def call(self, inputs):
    outerProduct = inputs[:,:, tf.newaxis] * inputs[:,tf.newaxis,:]
    return outerProduct

layer = QuadraticOuter()

def create_qlr_model(only_digits=True):
  """Create a Quadratic feature map Logistic Linear Model (for federated)

  When `only_digits=True`, the summary of returned model is

  Model: "sequential"

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    use_bias: A boolean that determines whether to use bias for the Logistic linear.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      QuadraticOuter(),
      tf.keras.layers.Flatten(input_shape=(784, 784)),
      AugmentedDense(10 if only_digits else 62, 
        name='dense',
        activation=tf.nn.softmax,
        **(projector_utils.get_lambd1_lambd2())),
  ])
  return model

def create_augmented_qlr_model_from_flags(only_digits=True):
  """
  Create AugmentedQLRModel with regularizer and projector
  """
  class AugmentedQLRModel(ProxModel):
    """
    Quadratic feature map Logistic Regression with regularizer and projector
    """
    def __init__(self, name=None, only_digits=True, projector=lambda _: None, regularizer=None):
      super(AugmentedQLRModel, self).__init__(name=name)
      self.projector = projector
      self.flattener_1 = tf.keras.layers.Flatten(
          input_shape=(28, 28, 1))
      self.quadratic_outer = QuadraticOuter()
      self.flattener_2 = tf.keras.layers.Flatten(
          input_shape=(784, 784))
      self.dense = AugmentedDense(
          10 if only_digits else 62, 
          name='dense',
          activation='softmax',
          kernel_regularizer=regularizer,
          **(projector_utils.get_lambd1_lambd2()))

    def call(self, inputs):
      return self.dense(
        self.flattener_2(
          self.quadratic_outer(
            self.flattener_1(inputs))))

  projector, regularizer = \
    projector_utils.build_proj_reg_from_flags()
 
  model = AugmentedQLRModel(
      only_digits=only_digits,
      projector=projector,
      regularizer=regularizer)

  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, 28, 28, 1))
  return model

## ------ 3. CNN model from original FedAvg paper -------
def create_cnn_model(only_digits=True):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper.

  When `only_digits=True`, the summary of returned model is
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 28, 28, 1)         0
  _________________________________________________________________
  conv2d (Conv2D)              (None, 28, 28, 32)        832
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
  _________________________________________________________________
  flatten (Flatten)            (None, 3136)              0
  _________________________________________________________________
  dense (Dense)                (None, 512)               1606144
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                5130
  =================================================================
  Total params: 1,663,370
  Trainable params: 1,663,370
  Non-trainable params: 0
  ```
  For `only_digits=False`, the last dense layer is slightly larger.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      AugmentedDense(512, 
        name = 'dense_1',
        activation=tf.nn.relu,
        **(projector_utils.get_lambd1_lambd2())),
      AugmentedDense(
        10 if only_digits else 62, 
        name = 'dense_2',
        activation=tf.nn.softmax,
        **(projector_utils.get_lambd1_lambd2())),
  ])

  return model

def create_augmented_cnn_model_from_flags(only_digits=True):
  """
  Create a ConvDropoutModelWithComposite
  """
  class AugmentedCNNModel(ProxModel):
    def __init__(self, only_digits=True, 
                projector=lambda _: None, regularizer=None):
      super(AugmentedCNNModel, self).__init__()
      self.projector = projector
      data_format = 'channels_last'
      self.conv1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        input_shape=(28, 28, 1),
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu
      )
      self.pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        data_format=data_format
      )
      self.conv2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu
      )
      self.pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same',
        data_format=data_format
      )
      self.flat = tf.keras.layers.Flatten()
      self.dense1 = AugmentedDense(512, 
        name = 'dense_1',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer,
        **(projector_utils.get_lambd1_lambd2()))
      self.dense2 = AugmentedDense(
        10 if only_digits else 62, 
        name = 'dense_2',
        kernel_regularizer=regularizer,
        activation=tf.nn.softmax,
        **(projector_utils.get_lambd1_lambd2()))
      
    def call(self, inputs, training=None, mask=None):
      x = self.conv1(inputs)
      x = self.pool1(x)
      x = self.conv2(x)
      x = self.pool2(x)
      x = self.flat(x)
      x = self.dense1(x)
      return self.dense2(x)
    
    def get_config(self):
      raise NotImplementedError

  projector, regularizer = \
    projector_utils.build_proj_reg_from_flags()

  model = AugmentedCNNModel(
      only_digits=only_digits,
      projector=projector,
      regularizer=regularizer)
  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, 28, 28, 1))
  return model

## ------ 4. 2nn Model -------
def create_2nn_model(only_digits=True, hidden_units=200):
  """Create a two hidden-layer fully connected neural network.

  When `only_digits=True`, the summary of returned model is

  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 784)               0
  _________________________________________________________________
  dense (Dense)                (None, 200)               157000
  _________________________________________________________________
  dense_1 (Dense)              (None, 200)               40200
  _________________________________________________________________
  dense_2 (Dense)              (None, 10)                2010
  =================================================================
  Total params: 199,210
  Trainable params: 199,210
  Non-trainable params: 0

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.
      We default to 200 units, which matches that in
      https://arxiv.org/abs/1602.05629.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      AugmentedDense(hidden_units, 
        name = 'dense_1',
        activation=tf.nn.relu,
        **(projector_utils.get_lambd1_lambd2())),
      AugmentedDense(hidden_units, 
        name = 'dense_2',
        activation=tf.nn.relu,
        **(projector_utils.get_lambd1_lambd2())),
      AugmentedDense(
        10 if only_digits else 62, 
        name = 'dense_3',
        activation=tf.nn.softmax,
        **(projector_utils.get_lambd1_lambd2())),
  ])

  return model

def create_augmented_2nn_model_from_flags(only_digits=True,
                                               hidden_units=200):
  """
  Create a Augmented2NNModel
  """
  class Augmented2NNModel(ProxModel):
    def __init__(self, name=None, only_digits=True, hidden_units=200, 
                  projector=lambda _: None, regularizer=None):
      super(Augmented2NNModel, self).__init__(name=name)
      self.projector = projector
      self.reshaper = tf.keras.layers.Reshape(
          input_shape=(28, 28, 1), target_shape=(28 * 28,))
      self.dense1 = AugmentedDense(hidden_units, 
                    name = 'dense_1',
                    activation=tf.nn.relu, 
                    kernel_regularizer=regularizer,
                    **(projector_utils.get_lambd1_lambd2()))
      self.dense2 = AugmentedDense(hidden_units, 
                    name = 'dense_2',
                    activation=tf.nn.relu, kernel_regularizer=regularizer,
                    **(projector_utils.get_lambd1_lambd2()))
      self.dense3 = AugmentedDense(
                    10 if only_digits else 62, activation=tf.nn.softmax,
                    name = 'dense_3',
                    kernel_regularizer=regularizer,
                    **(projector_utils.get_lambd1_lambd2()))

    def call(self, inputs):
      return self.dense3(self.dense2(self.dense1(self.reshaper(inputs))))

  projector, regularizer = \
    projector_utils.build_proj_reg_from_flags()

  model = Augmented2NNModel(
      only_digits=only_digits,
      hidden_units=hidden_units,
      projector=projector,
      regularizer=regularizer)
  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, 28, 28, 1))
  return model

