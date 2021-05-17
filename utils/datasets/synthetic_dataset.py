# Copyright 2019, The TensorFlow Federated Authors.
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
"""Library for loading and preprocessing any synthetic (h5) training and testing data."""
import pathlib

from typing import Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

TEST_BATCH_SIZE = 500
MAX_CLIENT_DATASET_SIZE = 418

def reshape_element(element):
  # need this to remove keys of orderdict
  return (element['x'], element['y'])


def load_data(dataset_name, data_dir_str=None):
  """
  load data with h5 cache
  """
  if data_dir_str is None:
    dir_path = pathlib.Path.home() / '.keras' / 'datasets'
  else:
    dir_path = pathlib.Path(data_dir_str)

  train_data_path = dir_path / (dataset_name + '_train.h5')
  test_data_path = dir_path / (dataset_name + '_test.h5')

  if train_data_path.exists() and test_data_path.exists():
    train_data = tff.simulation.hdf5_client_data.HDF5ClientData(
          str(train_data_path))
    test_data = tff.simulation.hdf5_client_data.HDF5ClientData(
          str(test_data_path))    
  else:
    raise NotImplementedError("File not found -- should be pre-downloaded")

  return train_data, test_data



def get_synthetic_datasets(dataset_name: str,
                        client_batch_size: int,
                        client_epochs_per_round: int,
                        max_batches_per_client: Optional[int] = -1):
  """Loads and preprocesses synthetic training and testing sets.

  Args:
    client_batch_size: Integer representing the batch size on the clients.
    client_epochs_per_round: Integer representing the number of epochs for which
      each client should perform training. This is done by repeating the
      dataset. If set to -1, the dataset is repeated indefinitely. In this case,
      the `max_batches_per_client` argument should be set to some positive
      integer, to ensure finite training time.
    max_batches_per_client: The maximum number of batches (of size
      `client_batch_size`) in the client dataset. This is enforced by using
      `tf.data.Dataset.take`. If set to -1 (the default value), then no maximum
      number of batches is enforced.

  Returns:
    train_data: An instance of a `tff.simulation.ClientData` representing the
      training data.
    test_data: An instance of a `tf.data.Dataset` representing the testing
      data.
  """

  if client_epochs_per_round == -1 and max_batches_per_client == -1:
    raise ValueError('Argument client_epochs_per_round is set to -1. If this is'
                     ' intended, then max_batches_per_client must be set to '
                     'some positive integer.')

  train_data, test_data = load_data(dataset_name)

  def preprocess_train_dataset(dataset):
    """Preprocessing function for the synthetic training dataset."""
    return (dataset
            # Shuffle according to the largest client dataset
            .shuffle(buffer_size=MAX_CLIENT_DATASET_SIZE)
            # Repeat to do multiple local epochs
            .repeat(client_epochs_per_round)
            # Batch to a fixed client batch size
            .batch(client_batch_size, drop_remainder=False)
            # Take a maximum number of batches
            .take(max_batches_per_client)
            # Preprocessing step
            .map(
                reshape_element,
                num_parallel_calls=tf.data.experimental.AUTOTUNE))

  def preprocess_test_dataset(dataset):
    """Preprocessing function for the EMNIST testing dataset."""
    return (dataset.batch(TEST_BATCH_SIZE, drop_remainder=False).map(
        reshape_element,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache())


  train_data = train_data.preprocess(preprocess_train_dataset)
  test_data = preprocess_test_dataset(
      test_data.create_tf_dataset_from_all_clients())
      # remove cache won't help

  return train_data, test_data


def get_centralized_synthetic_datasets(dataset_name: str,
                                    train_batch_size: int,
                             test_batch_size: Optional[int] = 500,
                             max_train_batches: Optional[int] = None,
                             max_test_batches: Optional[int] = None,
                             shuffle_train: Optional[bool] = True):
  """Loads and preprocesses centralized synthetic training and testing sets.

  Args:
    batch_size: An integer representing the batch size of the centralized
      training dateset.
    shuffle_train: A boolean indicating whether to shuffle the centralized train
      dataset.

  Returns:
    train_dataset: A `tf.data.Dataset` instance representing the training
      dataset.
    test_dataset: A `tf.data.Dataset` instance representing the test dataset.
  """
  train_data, test_data = load_data(dataset_name)

  def preprocess(dataset, batch_size, buffer_size=10000, shuffle_data=True):

    if shuffle_data:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    return (dataset.batch(batch_size).map(
        reshape_element,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )).cache()

  train_dataset = preprocess(
      train_data.create_tf_dataset_from_all_clients(),
      train_batch_size,
      shuffle_data=shuffle_train)
  test_dataset = preprocess(
      test_data.create_tf_dataset_from_all_clients(),
      test_batch_size,
      shuffle_data=False)

  if max_train_batches is not None and max_train_batches > 0:
    train_dataset = train_dataset.take(max_train_batches)
  if max_test_batches is not None and max_test_batches > 0:
    test_dataset = test_dataset.take(max_test_batches)

  return train_dataset, test_dataset
