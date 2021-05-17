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
import os

from absl import flags
from absl import logging

import tensorflow as tf
# import tensorflow_federated as tff

from utils import utils_impl

with utils_impl.record_hparam_flags():
  flags.DEFINE_integer(
      'intra_op', 0, 
      'tf.config.threading.set_intra_op_parallelism_thread')
  flags.DEFINE_integer(
      'inter_op', 0,
      'tf.config.threading.set_inter_op_parallelism_thread')
  flags.DEFINE_boolean('reference_executor', False, 'Uses Reference Executor')

FLAGS = flags.FLAGS

def set_threading_from_flags():
  """
  Limit threading
  """
  # if FLAGS.reference_executor:
  #     logging.info('Using Reference Executor')
  #     tff.framework.set_default_executor(tff.test.ReferenceExecutor())

  os.environ['OMP_NUM_THREADS'] = '1'
  tf.config.threading.set_inter_op_parallelism_threads(FLAGS.inter_op)
  tf.config.threading.set_intra_op_parallelism_threads(FLAGS.intra_op)

  logging.info(f'inter_op_parallelism_threads = {tf.config.threading.get_inter_op_parallelism_threads()}')
  logging.info(f'intra_op_parallelism_threads = {tf.config.threading.get_intra_op_parallelism_threads()}')
