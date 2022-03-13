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
"""Build a model for regularized logistic classification."""

# from abc import abstractmethod

# import numpy as np
from abc import abstractmethod
import tensorflow as tf
import sys
sys.path.insert(1, '/content/FCO-ICML21/')

from optimization.shared import projector_utils
from utils.models.augmented_dense import AugmentedDense

class ProxModel(tf.keras.Model):
  """
  Abstract Base Model that applies prox step after each gradient step
  """
  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # Forward pass
      # Compute the loss value
      # (the loss function is configured in `compile()`)

      # Regularization is handled separately !!!
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # also applies to proximal
    # proximator may accepts an optional kwarg lr
    self.proximator(trainable_vars, lr=self.optimizer.lr)

    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)

    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

  # @abstractmethod
  def get_config(self):
    pass

  # @abstractmethod
  def call(self, inputs, training=None, mask=None):
    pass

## ------ 1. Logistic Regression Model -------
def create_lr_model(num_attr):
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
      outputs for the larger dataset.z
    use_bias: A boolean that determines whether to use bias for the Logistic linear.

  Returns:
    A `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      AugmentedDense(1,
        name='dense',
        input_shape=(num_attr,),
        activation=tf.nn.sigmoid, 
        **(projector_utils.get_lambd1_lambd2())),
  ])
  return model

def create_prox_lr_model_from_flags(num_attr):
  """
  Create ProxLRModel with regularizer and projector
  (For centralized)
  """
  class ProxLRModel(ProxModel):
    """
    Logistic Regression with regularizer and projector
    """
    def __init__(self, name=None, proximator=lambda _: None,):
      super(ProxLRModel, self).__init__(name=name)
      self.proximator = proximator
      # _dict = projector_utils.get_lambd1_lambd2()
      # self.lambd1 = _dict['lambd1']
      # self.lambd2 = _dict['lambd2']

      self.dense = AugmentedDense(
          1,
          name ='dense',
          input_shape=(num_attr,),
          activation='sigmoid',
          **projector_utils.get_lambd1_lambd2())
          # lambd1 = self.lambd1,
          # lambd2 = self.lambd2)

    def call(self, inputs, training=None, mask=None):
      # self.add_metric(se)
      return self.dense(inputs)

  model = ProxLRModel(proximator=projector_utils.build_prox_from_flags())

  # Ref: https://stackoverflow.com/a/55909624/11662228
  model.build(input_shape=(None, num_attr))
  return model