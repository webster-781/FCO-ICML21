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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""

import collections
from typing import Any, Callable, Optional

import wandb
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff
import sys
sys.path.insert(1, '/content/FCO-ICML21/')

from optimization.emnist import federated_emnist
from optimization.bin_lr import federated_bin_lr
from optimization.lstsq import federated_lstsq
from optimization.nuclear import federated_nuclear


from optimization.shared import fed_avg_schedule
from optimization.shared import fed_dual_avg_schedule
from optimization.shared import optimizer_utils
from optimization.shared import projector_utils

from utils import threading_utils
from utils import utils_impl

_SUPPORTED_TASKS = [
    'emnist_cr', 'bin_lr', 'lstsq', 'nuclear'
]

with utils_impl.record_new_flags() as hparam_flags:
  # won't be added to the shared_flags
  flags.DEFINE_float('client_weight_pow', 1.0,
                     'How to weight in model delta. (num ** client_weight_pow)')
  flags.DEFINE_bool('use_subgrad', False, 'whether to use subgrad, default False, override dual-avg options' )
  flags.DEFINE_bool('use_dual_avg', True, 'whether to use dual averaging' )
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
# note: from iterative_process_builder
with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_float('client_sample_pow', 0.0,
                     'Sample prob of clients. (num ** client_sample_pow)')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_boolean(
      'write_metrics_with_bz2', True, 'Whether to use bz2 '
      'compression when writing output metrics to a csv file.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'rounds_per_train_eval', 10,
      'How often to evaluate the global model on the entire training dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'rounds_per_profile', 0,
      '(Experimental) How often to run the experimental TF profiler, if >0.')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

with utils_impl.record_hparam_flags() as emnist_cr_flags:
  # EMNIST CR flags
  pass

with utils_impl.record_hparam_flags() as bin_lr_flags:
  # Binary logistic regression flags
  flags.DEFINE_string(
    'bin_lr_dataset_name', None, 'The name of datasets')
  flags.DEFINE_integer(
    'bin_lr_num_attr', None, 'Number of attributes'
  )

with utils_impl.record_hparam_flags() as lstsq_flags:
  # Binary logistic regression flags
  flags.DEFINE_string(
    'lstsq_dataset_name', None, 'The name of datasets')
  flags.DEFINE_integer(
    'lstsq_num_attr', None, 'Number of attributes'
  )
  flags.DEFINE_integer(
    'lstsq_nnz_real', None, 'Number of ground truth non-zeros'
  )
  flags.DEFINE_float(
    'lstsq_nnz_cutoff', 1e-4, 'sparsity cutoff'
  )

with utils_impl.record_hparam_flags() as nuclear_flags:
  # Binary logistic regression flags
  flags.DEFINE_string(
    'nuclear_dataset_name', None, 'The name of datasets')
  flags.DEFINE_integer(
    'nuclear_n_row', None, 'Number of rows'
  )
  flags.DEFINE_integer(
    'nuclear_rank_real', None, 'Rank of ground truth'
  )
  flags.DEFINE_float(
    'nuclear_nnz_cutoff', 1e-4, 'sparsity cutoff'
  )

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
    emnist_cr=emnist_cr_flags,
    bin_lr=bin_lr_flags,
    lstsq=lstsq_flags,
    nuclear=nuclear_flags)

TASK_FLAG_PREFIXES = collections.OrderedDict(
    emnist_cr='emnist_cr',
    bin_lr='bin_lr',
    lstsq='lstsq',
    nuclear='nuclear')


def _get_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  return hparam_dict


def _get_task_args():
  """Returns an ordered dictionary of task-specific arguments.

  This method returns a dict of (arg_name, arg_value) pairs, where the
  arg_name has had the task name removed as a prefix (if it exists), as well
  as any leading `-` or `_` characters.

  Returns:
    An ordered dictionary of (arg_name, arg_value) pairs.
  """
  task_name = FLAGS.task
  task_args = collections.OrderedDict()

  if task_name in TASK_FLAGS:
    task_flag_list = TASK_FLAGS[task_name]
    task_flag_dict = utils_impl.lookup_flag_values(task_flag_list)
    task_flag_prefix = TASK_FLAG_PREFIXES[task_name]
    for (key, value) in task_flag_dict.items():
      if key.startswith(task_flag_prefix):
        key = key[len(task_flag_prefix):].lstrip('_-')
      task_args[key] = value
  return task_args


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  threading_utils.set_threading_from_flags()

  if FLAGS.experiment_name is not None:
    experiment_name = FLAGS.experiment_name
    wandb.init(config=FLAGS, sync_tensorboard=True,name=experiment_name)
  else:
    wandb.init(config=FLAGS,sync_tensorboard=True)
    wandb.run.save()
    wandb.run.name = FLAGS.task + "_f_" + str(wandb.run.name) + "_" + str(wandb.run.id)
    wandb.run.save()
    experiment_name = wandb.run.name

  # iterative_process_builder.py
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  client_mirror, server_mirror = projector_utils.build_mirror_fn_from_flags()

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
      client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor providing the weight
        in the federated average of model deltas. If not provided, the default
        is the total number of examples processed on device.

    Returns:
      A `tff.templates.IterativeProcess`.
    """
    if FLAGS.use_dual_avg:
      # Federated Dual Averaging
      process_builder = fed_dual_avg_schedule.build_fed_dual_avg_process
    else:
      # Federated Mirror Descent
      process_builder = fed_avg_schedule.build_fed_avg_process

    return process_builder(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        client_lr=client_lr_schedule,
        client_mirror=client_mirror,
        server_optimizer_fn=server_optimizer_fn,
        server_lr=server_lr_schedule,
        server_mirror=server_mirror,
        client_weight_fn=client_weight_fn,
        client_weight_pow=FLAGS.client_weight_pow)

  shared_args = utils_impl.lookup_flag_values(shared_flags)
  shared_args['iterative_process_builder'] = iterative_process_builder
  shared_args['experiment_name'] = experiment_name
  task_args = _get_task_args()
  hparam_dict = _get_hparam_flags()

  if FLAGS.task == 'emnist_cr':
    run_federated_fn = federated_emnist.run_federated
  elif FLAGS.task == 'bin_lr':
    run_federated_fn = federated_bin_lr.run_federated
  elif FLAGS.task == 'lstsq':
    run_federated_fn = federated_lstsq.run_federated
  elif FLAGS.task == 'nuclear':
    run_federated_fn = federated_nuclear.run_federated
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

  run_federated_fn(**shared_args, **task_args, hparam_dict=hparam_dict)


if __name__ == '__main__':
  app.run(main)
