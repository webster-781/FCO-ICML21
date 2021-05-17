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
"""Runs centralized training on various tasks with different optimizers.

The tasks, optimizers, and hyperparameters are all governed via flags. For more
details on the supported optimizers, see `shared/optimizer_utils.py`. For more
details on how the training loop is performed, see
`shared/centralized_training_loop.py`.
"""

import collections

from absl import app
from absl import flags

import wandb

from optimization.shared import optimizer_utils
from optimization.emnist import centralized_emnist
from optimization.bin_lr import centralized_bin_lr

from utils import threading_utils
from utils import utils_impl

_SUPPORTED_TASKS = [
    'emnist_cr', 'bin_lr'
]

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized_opt',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 32,
                       'Size of batches for training and eval.')
  flags.DEFINE_integer('decay_epochs', None, 'Number of epochs before decaying '
                       'the learning rate.')
  # delay epochs = 60?
  flags.DEFINE_float('lr_decay', None, 'How much to decay the learning rate by'
                     ' at each stage.')
  # lr decay = 0.2?

# with utils_impl.record_hparam_flags() as emnist_cr_flags:
#   pass

# with utils_impl.record_hparam_flags() as bin_lr_flags:
  # Binary logistic regression flags
  flags.DEFINE_string(
    'bin_lr_dataset_name', None, 'The name of datasets')
  flags.DEFINE_integer(
    'bin_lr_num_attr', None, 'Number of attributes'
  )
  # # emnist flag moves to emnist_models.py
  # # Stack Overflow LR flags
  # flags.DEFINE_integer('so_lr_vocab_tokens_size', 10000,
  #                      'Vocab tokens size used.')
  # flags.DEFINE_integer('so_lr_vocab_tags_size', 500, 'Vocab tags size used.')
  # flags.DEFINE_integer(
  #     'so_lr_num_validation_examples', 10000, 'Number of examples '
  #     'to use from test set for per-round validation.')

FLAGS = flags.FLAGS

# TASK_FLAGS = collections.OrderedDict(
#     emnist_cr=emnist_cr_flags,
#     bin_lr=bin_lr_flags)

# TASK_FLAG_PREFIXES = collections.OrderedDict(
#     emnist_cr='emnist_cr',
#     bin_lr='bin_lr')



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  threading_utils.set_threading_from_flags()

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  hparams_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  if FLAGS.experiment_name is not None:
    experiment_name = FLAGS.experiment_name
    wandb.init(config=FLAGS, sync_tensorboard=False,name=experiment_name)
  else:
    wandb.init(config=FLAGS,sync_tensorboard=False)
    wandb.run.save()
    wandb.run.name = FLAGS.task + "_c_" + str(wandb.run.name) + "_" + str(wandb.run.id)
    wandb.run.save()
    experiment_name = wandb.run.name

  common_args = collections.OrderedDict([
      ('optimizer', optimizer),
      ('experiment_name', experiment_name),
      ('root_output_dir', FLAGS.root_output_dir),
      ('num_epochs', FLAGS.num_epochs),
      ('batch_size', FLAGS.batch_size),
      ('decay_epochs', FLAGS.decay_epochs),
      ('lr_decay', FLAGS.lr_decay),
      ('hparams_dict', hparams_dict),
  ])

  if FLAGS.task == 'emnist_cr':
    centralized_emnist.run_centralized(**common_args)
  
  elif FLAGS.task == 'bin_lr':
    bin_lr_flags = collections.OrderedDict()
    for flag_name in FLAGS:
      if flag_name.startswith('bin_lr_'):
        bin_lr_flags[flag_name[7:]] = FLAGS[flag_name].value
    centralized_bin_lr.run_centralized(**common_args, **bin_lr_flags)

  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))


if __name__ == '__main__':
  app.run(main)
