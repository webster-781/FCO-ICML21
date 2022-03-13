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
"""Build a model for Least-Squares Regression."""

# from abc import abstractmethod
import sys
sys.path.insert(1, '/content/FCO-ICML21/')

import tensorflow as tf
from optimization.shared import projector_utils


# class ProxModel(tf.keras.Model):
#   """
#   ABC class for Model with Projector
#   """
#   def train_step(self, data):
#     x, y = data
#     with tf.GradientTape() as tape:
#       y_pred = self(x, training=True)  # Forward pass
#       # Compute the loss value
#       # (the loss function is configured in `compile()`)
#       loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

#     # Compute gradients
#     trainable_vars = self.trainable_variables
#     gradients = tape.gradient(loss, trainable_vars)
#     # Update weights
#     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#     self.projector(trainable_vars)
#     # Update metrics (includes the metric that tracks the loss)
#     self.compiled_metrics.update_state(y, y_pred)

#     # Return a dict mapping metric names to current value
#     return {m.name: m.result() for m in self.metrics}

#   @abstractmethod
#   def get_config(self):
#     pass

#   @abstractmethod
#   def call(self, inputs, training=None, mask=None):
#     pass


class AugmentedDense(tf.keras.layers.Dense):
  # this is not functional in TFF... from_keras_model won't accept subclass model
  def __init__(self, units, nnz_real=None, nnz_cutoff=1e-4, lambd1=0.0, lambd2=0.0, **kwargs):
    self.nnz_cutoff = nnz_cutoff
    self.lambd1 = lambd1
    self.lambd2 = lambd2
    self.nnz_real = nnz_real
    super(AugmentedDense, self).__init__(units, **kwargs)
  def call(self, inputs, training=None):
    if not training:
      l1 = tf.norm(self.kernel, ord=1)
      l2 = tf.norm(self.kernel)
      self.add_metric(l1, name='l1', aggregation='mean')
      self.add_metric(l2, name='l2', aggregation='mean')
      reg_loss = self.lambd1 * l1 + self.lambd2 * (l2 ** 2.0)

      if self.nnz_real is not None:
        kernel_nnz_boolean = (tf.math.abs(self.kernel) > self.nnz_cutoff)
        nnz_pos = tf.math.count_nonzero(kernel_nnz_boolean[:self.nnz_real,:])
        nnz_pred = tf.math.count_nonzero(kernel_nnz_boolean)

        precision = nnz_pos / nnz_pred
        recall = nnz_pos / self.nnz_real
        f1_score = 2*precision*recall/(precision+recall)

        # need aggregation due to https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/base_layer_v1.py#L1896
        self.add_metric(tf.cast(nnz_pred, tf.float32), name='l0', aggregation='mean')
        self.add_metric(f1_score, name='F1', aggregation='mean')
        self.add_metric(precision, name='precision', aggregation='mean')
        self.add_metric(recall, name='recall', aggregation='mean')

    return super(AugmentedDense, self).call(inputs)

def create_lstsq_model(num_attr, nnz_real=None, nnz_cutoff=1e-4):
  """Create a Least Squares Base Model (for federated)

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
      AugmentedDense(1, 
        name='dense',
        input_shape=(num_attr,),  
        nnz_real=nnz_real,
        nnz_cutoff=nnz_cutoff,
        **(projector_utils.get_lambd1_lambd2()))
  ])
  return model