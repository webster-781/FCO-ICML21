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
"""Shared library for setting up federated training experiments."""

import collections
import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**(31) - 1
MLCG_MULTIPLIER = 16807


# TODO(b/143440780): Create more comprehensive tuple conversion by adding an
# explicit namedtuple checking utility.
def convert_to_tuple_dataset(dataset):
  """Converts a dataset to one where elements have a tuple structure.

  Args:
    dataset: A `tf.data.Dataset` where elements either have a mapping
      structure of format {"x": <features>, "y": <labels>}, or a tuple-like
        structure of format (<features>, <labels>).

  Returns:
    A `tf.data.Dataset` object where elements are tuples of the form
    (<features>, <labels>).

  """
  example_structure = dataset.element_spec
  if isinstance(example_structure, collections.Mapping):
    # We assume the mapping has `x` and `y` keys.
    convert_map_to_tuple = lambda example: (example['x'], example['y'])
    try:
      return dataset.map(convert_map_to_tuple)
    except:
      raise ValueError('For datasets with a mapping structure, elements must '
                       'have format {"x": <features>, "y": <labels>}.')
  elif isinstance(example_structure, tuple):

    if hasattr(example_structure, '_fields') and isinstance(
        example_structure._fields, collections.Sequence) and all(
            isinstance(f, str) for f in example_structure._fields):
      # Dataset has namedtuple structure
      convert_tuplelike_to_tuple = lambda x: (x[0], x[1])
    else:
      # Dataset does not have namedtuple structure
      convert_tuplelike_to_tuple = lambda x, y: (x, y)

    try:
      return dataset.map(convert_tuplelike_to_tuple)
    except:
      raise ValueError('For datasets with tuple-like structure, elements must '
                       'have format (<features>, <labels>).')
  else:
    raise ValueError(
        'Expected evaluation dataset to have elements with a mapping or '
        'tuple-like structure, found {} instead.'.format(example_structure))


def build_evaluate_fn(
    eval_dataset: tf.data.Dataset, model_builder: Callable[[], tf.keras.Model],
    loss_builder: Callable[[], tf.keras.losses.Loss],
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]]
) -> Callable[[tff.learning.ModelWeights], Dict[str, Any]]:
  """Builds an evaluation function for a given model and test dataset.

  The evaluation function takes as input a fed_avg_schedule.ServerState, and
  computes metrics on a keras model with the same weights.

  Args:
    eval_dataset: A `tf.data.Dataset` object. Dataset elements should either
      have a mapping structure of format {"x": <features>, "y": <labels>}, or a
        tuple structure of format (<features>, <labels>).
    model_builder: A no-arg function that returns a `tf.keras.Model` object.
    loss_builder: A no-arg function returning a `tf.keras.losses.Loss` object.
    metrics_builder: A no-arg function that returns a list of
      `tf.keras.metrics.Metric` objects.

  Returns:
    A function that take as input a `tff.learning.ModelWeights` and returns
    a dict of (name, value) pairs for each associated evaluation metric.
  """

  def compiled_eval_keras_model():
    model = model_builder()
    model.compile(
        loss=loss_builder(),
        optimizer=tf.keras.optimizers.SGD(),  # Dummy optimizer for evaluation
        metrics=metrics_builder())
    return model

  eval_tuple_dataset = convert_to_tuple_dataset(eval_dataset)

  def evaluate_fn(reference_model: tff.learning.ModelWeights) -> Dict[str, Any]:
    """Evaluation function to be used during training."""

    if not isinstance(reference_model, tff.learning.ModelWeights):
      raise TypeError('The reference model used for evaluation must be a'
                      '`tff.learning.ModelWeights` instance.')

    keras_model = compiled_eval_keras_model()
    reference_model.assign_weights_to(keras_model)
    logging.info('Evaluating the current model')
	
    eval_metrics = keras_model.evaluate(eval_tuple_dataset, verbose=0)
    metrics_dict = dict(zip(keras_model.metrics_names, eval_metrics))

    l0 = 0.0
    max_l1 = 0.0
    max_l2 = 0.0
    reg_loss = 0.0

    for key, val in (metrics_dict.items()):
      # val = val.numpy()
      if isinstance(key, str):
        if key.endswith("l0"):
          l0 += val
        elif key.endswith("l1"):
          max_l1 = max(max_l1, val)
        elif key.endswith("l2"):
          max_l2 = max(max_l2, val)
        elif key.endswith("reg_loss"):
          reg_loss += val

    metrics_dict['l0'] = l0
    metrics_dict['max_l1'] = max_l1
    metrics_dict['max_l2'] = max_l2
    metrics_dict['tot_loss'] = metrics_dict['loss'] + reg_loss

    return metrics_dict
	
  return evaluate_fn


def build_sample_fn(
    a: Union[Sequence[Any], int],
    size: int,
    replace: bool = False,
    p: Union[Sequence[Any], None] = None,
    random_seed: Optional[int] = None) -> Callable[[int], np.ndarray]:
  """Builds the function for sampling from the input iterator at each round.

  Args:
    a: A 1-D array-like sequence or int that satisfies np.random.choice.
    size: The number of samples to return each round.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).
    p: The probabilities associated with each entry in a. If not given the 
      sample assumes a uniform distribution over all entries in a. 
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process.

  Returns:
    A function which returns a list of elements from the input iterator at a
    given round round_num.
  """
  if isinstance(random_seed, int):
    mlcg_start = np.random.RandomState(random_seed).randint(1, MLCG_MODULUS - 1)

    def get_pseudo_random_int(round_num):
      return pow(MLCG_MULTIPLIER, round_num,
                 MLCG_MODULUS) * mlcg_start % MLCG_MODULUS

  def sample(round_num, random_seed):
    if isinstance(random_seed, int):
      random_state = np.random.RandomState(get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()
    return random_state.choice(a, size=size, replace=replace, p=p)

  return functools.partial(sample, random_seed=random_seed)


def build_client_datasets_fn(
    train_dataset: tff.simulation.ClientData,
    train_clients_per_round: int,
    client_sample_pow: float = 0.0,
    random_seed: Optional[int] = None
) -> Callable[[int], List[tf.data.Dataset]]:
  """Builds the function for generating client datasets at each round.

  The function samples a number of clients (without replacement within a given
  round, but with replacement across rounds) and returns their datasets.

  Args:
    train_dataset: A `tff.simulation.ClientData` object.
    train_clients_per_round: The number of client participants in each round.
    random_seed: If random_seed is set as an integer, then we use it as a random
      seed for which clients are sampled at each round. In this case, we set a
      random seed before sampling clients according to a multiplicative linear
      congruential generator (aka Lehmer generator, see 'The Art of Computer
      Programming, Vol. 3' by Donald Knuth for reference). This does not affect
      model initialization, shuffling, or other such aspects of the federated
      training process. Note that this will alter the global numpy random seed.

  Returns:
    A function which returns a list of `tf.data.Dataset` objects at a
    given round round_num.
  """

  n_samples = np.array([tf.data.experimental.cardinality(
      train_dataset.create_tf_dataset_for_client(client_id)
      ) for client_id in train_dataset.client_ids])
  unnormalized_p = n_samples ** client_sample_pow
  p = unnormalized_p / sum(unnormalized_p)

  sample_clients_fn = build_sample_fn(
      train_dataset.client_ids,
      size=train_clients_per_round,
      replace=False,
      p=p,
      random_seed=random_seed)
  # else:
  #   sample_clients_fn = build_sample_fn(
  #       train_dataset.client_ids,
  #       size=train_clients_per_round,
  #       replace=False,
  #       random_seed=random_seed)

  def client_datasets(round_num):
    sampled_clients = sample_clients_fn(round_num)
    return [
        train_dataset.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]

  return client_datasets
