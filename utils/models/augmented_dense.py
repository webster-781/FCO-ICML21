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
import tensorflow as tf

class AugmentedDense(tf.keras.layers.Dense):
  # A Dense layer that RECORDS l1 and l2 norm.
  # from_keras will not accept subclass model, but subclassed layer is OK
  def __init__(self, units, nnz_cutoff=1e-4, lambd1=0.0, lambd2=0.0, **kwargs):
    self.nnz_cutoff = nnz_cutoff
    self.lambd1 = lambd1
    self.lambd2 = lambd2
    super(AugmentedDense, self).__init__(units, **kwargs)
  def call(self, inputs, training=None):
    if (not training):
      # kernel_nnz_boolean = (tf.math.abs(self.kernel) > self.nnz_cutoff)
      # nnz_pred = tf.math.count_nonzero(kernel_nnz_boolean)
      l0 = tf.math.count_nonzero(tf.math.abs(self.kernel) > self.nnz_cutoff)
      l1 = tf.norm(self.kernel, ord=1)
      l2 = tf.norm(self.kernel)
      reg_loss = self.lambd1 * l1 + self.lambd2 * (l2 ** 2.0)
      
      # need aggregation due to https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/base_layer_v1.py#L1896
      # Ref: https://keras.io/api/layers/base_layer/
      # Ref: https://github.com/tensorflow/tensorflow/blob/9a0e4701dfd2817e90cead366892777c0b77ee97/tensorflow/python/keras/engine/base_layer.py#L1613
      self.add_metric(tf.cast(l0, tf.float32), name=self.name+'_l0', aggregation='mean')
      self.add_metric(l1, name=self.name+'_l1', aggregation='mean')
      self.add_metric(l2, name=self.name+'_l2', aggregation='mean')
      self.add_metric(reg_loss, name=self.name+'_reg_loss', aggregation='mean')

    return super(AugmentedDense, self).call(inputs)

