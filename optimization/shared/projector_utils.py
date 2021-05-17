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
# Lint as: python3

import functools
import wandb

import tensorflow as tf

from absl import flags
from utils import utils_impl
from optimization.shared.simplex_projection import euclidean_proj_l1ball

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum('composite', 'none', ['none', 'l1-proj', 'l2-proj', 
                                          'l1-reg', 'l2-reg', 'nuc-reg'],
                    'which composite mode to use')

  flags.DEFINE_float('ball_size', 1.0,
                     'size of ball being projected to (if l1-proj or l2-proj) mode')

  flags.DEFINE_float('lambd', 1.0,
                     'regularization strength (if l1-reg or l2-reg)')

  flags.DEFINE_bool('use_client_mirror', True,
                    'whether to use client mirror. server mirror is adjusted accordingly')

FLAGS = flags.FLAGS

def get_lambd1_lambd2():
  if FLAGS.composite == 'l1-reg':
    return {'lambd1': FLAGS.lambd, 'lambd2': 0.0}
  elif FLAGS.composite == 'l2-reg':
    return {'lambd2': FLAGS.lambd, 'lambd1': 0.0}
  else:
    return dict()

def get_lambd():
  return {'lambd': FLAGS.lambd}


def proj_l1(list_of_trainable, ball_size=1, **kwargs):
  """
  Project each kernel to l1 ball, in place
  """
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      flattened_tensor = tf.reshape(variable, [-1])
      projected_tensor = euclidean_proj_l1ball(flattened_tensor, ball_size)
      variable.assign(tf.reshape(projected_tensor, variable.shape))

def proj_l2(list_of_trainable, ball_size=1, **kwargs):
  """
  Project each kernel to l2 ball, in place
  """
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      scaler = tf.math.maximum(tf.norm(variable) / ball_size, 1)
      variable.assign(variable / scaler)

def subgrad_l1(list_of_trainable, lambd=None, lr=None):
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      variable.assign(variable - lambd * lr * tf.sign(variable))


def prox_l1(list_of_trainable, lambd=None, lr=None):
    # Soft threshold ref: https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      variable.assign(
        soft_threshold(variable, lambd*lr)
        # tf.sign(variable) * \
        #   tf.maximum(tf.abs(variable) - lambd * lr, 0.)
      )

def prox_l2(list_of_trainable, lambd=None, lr=None):
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      variable.assign(variable / (1 + 2 * lambd * lr))


def soft_threshold(var, strength):
  return tf.sign(var) * \
          tf.maximum(tf.abs(var) - strength, 0.)


def prox_nuc(list_of_trainable, n_row=None, lambd=None, lr=None):
    # Soft threshold ref: https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator
  for variable in list_of_trainable:
    if 'dense' in variable.name and  'kernel' in variable.name:
      kernel_matrix = tf.reshape(variable, (n_row, n_row))
      s, u, v, = tf.linalg.svd(kernel_matrix)
      s = soft_threshold(s, lambd*lr)
      kernel_matrix = u @ tf.linalg.diag(s) @ tf.transpose(v)
      variable.assign(tf.reshape(kernel_matrix, (n_row*n_row,1)))



def build_mirror_fn_from_flags():
  """
  Get (client_mirror, server_mirror) from flags.
  """
  identity = lambda _, **kwargs: None

  if FLAGS.composite == 'none':
    client_mirror = identity
    server_mirror = identity
  elif FLAGS.composite.endswith("proj"):
    # projector mode, server_mirror always on
    projector = functools.partial(proj_l1, ball_size=FLAGS.ball_size) \
      if FLAGS.composite == 'l1-proj'  \
      else functools.partial(proj_l2, ball_size=FLAGS.ball_size)

    server_mirror = projector
    client_mirror = projector if FLAGS.use_client_mirror else identity

  elif FLAGS.composite.endswith("reg"):
    if not FLAGS.use_subgrad:
      if FLAGS.composite == 'l1-reg':
        mirror = functools.partial(prox_l1, lambd=FLAGS.lambd)
      # elif FLAGS.composite == "l2-reg":
      #   mirror = functools.partial(prox_l2, lambd=FLAGS.lambd)
      elif FLAGS.composite == "nuc-reg":
        mirror = functools.partial(prox_nuc, lambd=FLAGS.lambd, n_row=FLAGS.nuclear_n_row) 
      else:
        raise NotImplementedError()

      # server is always mirror
      # client is mirror iff use_client_mirror
      server_mirror = mirror
      client_mirror = mirror if FLAGS.use_client_mirror else identity
    else: # use subgrad
      if FLAGS.composite == 'l1-reg':
        client_mirror = functools.partial(subgrad_l1, lambd=FLAGS.lambd)
      else:
        raise NotImplementedError()
      
      server_mirror = identity

  else:
    raise NotImplementedError()
  return client_mirror, server_mirror


def build_prox_from_flags():
  """ build prox model for centralized """
  identity = lambda _: None

  if FLAGS.composite == 'none':
    return identity
  if FLAGS.composite == 'l1-reg':
    return functools.partial(prox_l1, lambd=FLAGS.lambd)
  elif FLAGS.composite == 'l2-reg':
    return functools.partial(prox_l2, lambd=FLAGS.lambd)
  elif FLAGS.composite == 'l1-proj':
    return functools.partial(proj_l1, ball_size=FLAGS.ball_size)
  elif FLAGS.composite == 'l2-proj':
    return functools.partial(proj_l2, ball_size=FLAGS.ball_size)
  else:
    raise NotImplementedError

def build_proj_reg_from_flags():
  """ build proj model for centralized """
  identity = lambda _: None

  if FLAGS.composite == 'none':
    return identity, None
  if FLAGS.composite == 'l1-reg':
    return identity, tf.keras.regularizers.l1(FLAGS.lambd)
  elif FLAGS.composite == 'l2-reg':
    return identity, tf.keras.regularizers.l2(FLAGS.lambd)
  elif FLAGS.composite == 'l1-proj':
    return functools.partial(proj_l1, ball_size=FLAGS.ball_size), None
  elif FLAGS.composite == 'l2-proj':
    return functools.partial(proj_l2, ball_size=FLAGS.ball_size), None
  else:
    raise NotImplementedError

class RecordWeightNorm(tf.keras.callbacks.Callback):
  """
  Record the l1 and l2 norm of the weights via wandb
  """
  def __init__(self):
    super(RecordWeightNorm, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    wandb.log({
        'max_kernel_l2_norm': max(
          [tf.norm(variable).numpy() for variable in self.model.variables
            if 'dense' in variable.name and 'kernel' in variable.name]),
        'epoch': epoch
    })
    wandb.log({
        'max_kernel_l1_norm': max(
          [tf.norm(variable, 1).numpy() for variable in self.model.variables
            if 'dense' in variable.name and 'kernel' in variable.name]),
        'epoch': epoch
    })

