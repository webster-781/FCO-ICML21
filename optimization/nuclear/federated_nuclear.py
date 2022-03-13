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
"""Federated EMNIST character recognition library using TFF."""

import functools
from typing import Callable, Optional
import sys
sys.path.insert(1, '/content/FCO-ICML21/')

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import training_loop
from utils import training_utils
from utils.datasets import synthetic_dataset
from utils.models import nuclear_models

FLAGS = flags.FLAGS

def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    client_sample_pow: Optional[float] = 0.0,
    max_batches_per_client: Optional[int] = -1,
    client_datasets_random_seed: Optional[int] = None,
    total_rounds: Optional[int] = 1500,
    experiment_name: Optional[str] = 'federated_nuclear',
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    max_eval_batches: Optional[int] = None,
    dataset_name: Optional[str] = None,
    n_row: Optional[int] = None,
    rank_real: Optional[int] = None,
    nnz_cutoff: Optional[float] = 1e-4,
    **kwargs):
  """Runs an iterative process on the EMNIST character recognition task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Moreover, the server state must have an attribute `model` of type
  `tff.learning.ModelWeights`.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, and
      returns a `tff.templates.IterativeProcess`. The `model_fn` must return a
      `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    max_batches_per_client: An optional int specifying the number of batches
      taken by each client at each round. If `-1`, the entire client dataset is
      used.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    model: A string specifying the model used for character recognition.
      Can be one of `cnn` and `2nn`, corresponding to a CNN model and a densely
      connected 2-layer model (respectively).
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    max_eval_batches: If set to a positive integer, evaluation datasets are
      capped to at most that many batches. If set to None or a nonpositive
      integer, the full evaluation datasets are used.
    **kwargs: Additional arguments configuring the training loop. For details
      on supported arguments, see
      `federated_research/utils/training_utils.py`.
  """
  train_ds, test_ds = synthetic_dataset.get_synthetic_datasets(
      dataset_name,
      client_batch_size,
      client_epochs_per_round,
      max_batches_per_client=max_batches_per_client,
      )

  centralized_train_ds, test_ds = \
    synthetic_dataset.get_centralized_synthetic_datasets(
      dataset_name=dataset_name,
      train_batch_size=client_batch_size,
      max_test_batches=max_eval_batches,
      )

  input_spec = train_ds.element_type_structure

  model_builder = functools.partial(
        nuclear_models.create_nuclear_model,
        n_row=n_row,
        rank_real=rank_real,
        nnz_cutoff=nnz_cutoff,
        )

  loss_builder = tf.keras.losses.MeanSquaredError
  metrics_builder = lambda: []

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  training_process = iterative_process_builder(tff_model_fn)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      train_dataset=train_ds,
      train_clients_per_round=clients_per_round,
      client_sample_pow=client_sample_pow,
      random_seed=client_datasets_random_seed)

  evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=test_ds,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  train_evaluate_fn = training_utils.build_evaluate_fn(
      eval_dataset=centralized_train_ds,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=evaluate_fn,
      train_eval_fn=train_evaluate_fn,
      test_fn=evaluate_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
