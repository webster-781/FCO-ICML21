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
"""Build a model for nuclear-regularized Least-Squares Regression."""

# from abc import abstractmethod

import tensorflow as tf
from optimization.shared import projector_utils

class NuclearDense(tf.keras.layers.Dense):
  # this is not functional in TFF... from_keras_model won't accept subclass model
  def __init__(self, units, n_row, rank_real=None, nnz_cutoff=1e-4, lambd=0.0, **kwargs):
    # n_col = n_row for simplicity
    self.n_row = n_row
    self.lambd = lambd
    self.rank_real = rank_real
    self.nnz_cutoff = nnz_cutoff
    super(NuclearDense, self).__init__(units, **kwargs)
    
  def call(self, inputs, training=None):
    if not training:

      kernel_matrix = tf.reshape(self.kernel, (self.n_row, self.n_row))
      singular_vals, _, _ = tf.linalg.svd(kernel_matrix)

      nuc = tf.reduce_sum(singular_vals)
      reg_loss = self.lambd * nuc
      rank = tf.math.count_nonzero(singular_vals > self.nnz_cutoff)

      self.add_metric(nuc, name=self.name+'_nuc', aggregation = 'mean')
      self.add_metric(reg_loss, name=self.name+'_reg_loss', aggregation='mean')
      self.add_metric(tf.cast(rank,tf.float32), name='rank', aggregation='mean')


      if self.rank_real is not None:
        ground_truth = tf.linalg.diag(
          [1.0]*self.rank_real+[0.0]*(self.n_row-self.rank_real))

        err_fro = tf.norm(ground_truth-kernel_matrix)

        # need aggregation due to https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/base_layer_v1.py#L1896
        self.add_metric(err_fro, name='err_fro', aggregation='mean')

    return super(NuclearDense, self).call(inputs)

def create_nuclear_model(n_row, rank_real=None, nnz_cutoff=1e-4):
  """Create a Least Squares Base Model (for federated) with nuclear regularization

  Args:
    use_bias: A boolean that determines whether to use bias for the Logistic linear.

  Returns:
    A `tf.keras.Model`.
  """

  model = tf.keras.models.Sequential([
      NuclearDense(1, 
        name='dense',
        input_shape=(n_row*n_row,),  
        n_row = n_row,
        rank_real=rank_real,
        nnz_cutoff=nnz_cutoff,
        **(projector_utils.get_lambd()))
  ])
  return model