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
"""An implementation of the FedAvg algorithm with learning rate schedules.

This is intended to be a somewhat minimal implementation of Federated
Averaging that allows for client and server learning rate scheduling.

The original FedAvg is based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import collections
from typing import Callable, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.tensorflow_libs import tensor_utils


# Convenience type aliases.
ModelBuilder = Callable[[], tff.learning.Model]
OptimizerBuilder = Callable[[float], tf.keras.optimizers.Optimizer]
ClientWeightFn = Callable[..., float]
LRScheduleFn = Callable[[Union[int, tf.Tensor]], Union[tf.Tensor, float]]


def _initialize_optimizer_vars(model: tff.learning.Model,
                               optimizer: tf.keras.optimizers.Optimizer):
  """Ensures variables holding the state of `optimizer` are created."""
  delta = tf.nest.map_structure(tf.zeros_like, _get_weights(model).trainable)
  model_weights = _get_weights(model)
  grads_and_vars = tf.nest.map_structure(lambda x, v: (x, v), delta,
                                         model_weights.trainable)
  optimizer.apply_gradients(grads_and_vars, name='server_update')
  assert optimizer.variables()


def _get_weights(model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tff.learning.ModelWeights.from_model(model)


@attr.s(eq=False, order=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  -   `model`: primal model (weights), A dictionary of the model's trainable and non-trainable weights (consistent with FedAvg)
  -   `dual_model`: dual model (weights),  dictionary of the model's trainable and non-trainable weights.
  -   `optimizer_state`: The server optimizer variables.
  -   `round_num`: The current training round, as a float.
  """
  model = attr.ib()
  dual_model_weights = attr.ib() # actually model_weights
  optimizer_state = attr.ib()
  elapsed_lr = attr.ib()
  round_num = attr.ib()
  # This is a float to avoid type incompatibility when calculating learning rate
  # schedules.


@tf.function
def server_update(primal_model, dual_model, server_optimizer, server_mirror, 
                  server_state, weights_delta, elapsed_lr_delta):
  """Updates `server_state` based on `weights_delta`, increase the round number.

  Args:
    model: A `tff.learning.Model`.
	dual_model: A `tff.learning.Model` for dual weights.
    server_optimizer: A `tf.keras.optimizers.Optimizer`.
    server_state: A `ServerState`, the state to be updated.
    weights_delta: An update to the trainable variables of the model.

  Returns:
    An updated `ServerState`.
  """
  dual_model_weights = _get_weights(dual_model)
  # server state hold dual model
  tff.utils.assign(dual_model_weights, server_state.dual_model_weights)
  # Server optimizer variables must be initialized prior to invoking this
  tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

  weights_delta, has_non_finite_weight = (
      tensor_utils.zero_all_if_any_non_finite(weights_delta))
  if has_non_finite_weight > 0:
    return server_state

  # Apply the update to the model. We must multiply weights_delta by -1.0 to
  # view it as a gradient that should be applied to the server_optimizer.
  grads_and_vars = [
      (-1.0 * x, v) for x, v in zip(weights_delta, dual_model_weights.trainable)
  ]

  server_optimizer.apply_gradients(grads_and_vars)
  elapsed_lr = server_state.elapsed_lr + elapsed_lr_delta * server_optimizer.lr

  primal_model_weights = _get_weights(primal_model)
  tff.utils.assign(primal_model_weights, dual_model_weights)
  server_mirror(primal_model_weights.trainable, lr=elapsed_lr)

  # Create a new state based on the updated model.
  return tff.utils.update_state(
      server_state,
      model=primal_model_weights,
      dual_model_weights=dual_model_weights,
      optimizer_state=server_optimizer.variables(),
      elapsed_lr = elapsed_lr,
      round_num=server_state.round_num + 1.0)


@attr.s(eq=False, order=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated optimization.

  Fields:
  -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
  -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
  -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
  -   `optimizer_output`: Additional metrics or other outputs defined by the
      optimizer.
  """
  weights_delta = attr.ib()
  client_weight = attr.ib()
  elapsed_lr_delta = attr.ib() # necessary for dual averaging
  model_output = attr.ib()
  optimizer_output = attr.ib()


def create_client_update_fn():
  """Returns a tf.function for the client_update.

  This "create" fn is necesessary to prevent
  "ValueError: Creating variables on a non-first call to a function decorated
  with tf.function" errors due to the client optimizer creating variables. This
  is really only needed because we test the client_update function directly.
  """

  @tf.function
  def client_update(primal_model,
                    dual_model,
                    dataset,
                    dual_initial_weights,
                    client_optimizer,
                    client_mirror,
                    elapsed_lr,
                    client_weight_fn=None,
                    client_weight_pow=1):
    """Updates client model.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      dual_initial_weights: A `tff.learning.Model.weights` from server.
      client_optimizer: A `tf.keras.optimizer.Optimizer` object.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor that provides the
        weight in the federated average of model deltas. If not provided, the
        default is the total number of examples processed on device.

    Returns:
      A 'ClientOutput`.
    """

    primal_model_weights = _get_weights(primal_model)
    dual_model_weights = _get_weights(dual_model)
    new_elapsed_lr = elapsed_lr

    tff.utils.assign(dual_model_weights, dual_initial_weights)
    num_examples = tf.constant(0, dtype=tf.int32)

    for batch in dataset:
      # assign dual to primal
      tff.utils.assign(primal_model_weights, dual_model_weights)

      # apply (in place) projector to primal model
      client_mirror(primal_model_weights.trainable, lr=new_elapsed_lr)

      # tape gradients
      with tf.GradientTape() as tape:
        output = primal_model.forward_pass(batch)
      
      grads = tape.gradient(output.loss, primal_model_weights.trainable)

      # zip gradient with DUAL trainable
      grads_and_vars = zip(grads, dual_model_weights.trainable)

      # apply gradients (to dual)
      client_optimizer.apply_gradients(grads_and_vars)

      num_examples += tf.shape(output.predictions)[0]
      new_elapsed_lr += client_optimizer.lr

    aggregated_outputs = primal_model.report_local_outputs()
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          dual_model_weights.trainable,
                                          dual_initial_weights.trainable)
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(float(num_examples) ** float(client_weight_pow), tf.float32)
    else:
      client_weight = client_weight_fn(aggregated_outputs)

    return ClientOutput(
        weights_delta, client_weight,
        new_elapsed_lr - elapsed_lr, aggregated_outputs,
        collections.OrderedDict([('num_examples', num_examples)]))

  return client_update


def build_server_init_fn(
    model_fn: ModelBuilder,
    server_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer]):  
  """Builds a `tff.tf_computation` that returns the initial `ServerState`.

  The attributes `ServerState.dual_model` and `ServerState.optimizer_state` are
  initialized via their constructor functions. The attribute
  `ServerState.round_num` is set to 0.0.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.

  Returns:
    A `tff.tf_computation` that returns initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    server_optimizer = server_optimizer_fn()
    primal_model = model_fn()
    dual_model = model_fn()
    _initialize_optimizer_vars(dual_model, server_optimizer)
    return ServerState(
        model=_get_weights(primal_model),
        dual_model_weights=_get_weights(dual_model),
        optimizer_state=server_optimizer.variables(),
        elapsed_lr=0.0,
        round_num=0.0)

  return server_init_tf


def build_fed_dual_avg_process(
    model_fn: ModelBuilder,
    client_optimizer_fn: OptimizerBuilder,
    client_lr: Union[float, LRScheduleFn] = 0.1,
    client_mirror=(lambda _: None),
    server_optimizer_fn: OptimizerBuilder = tf.keras.optimizers.SGD,
    server_lr: Union[float, LRScheduleFn] = 1.0,
    server_mirror=(lambda _: None),
    client_weight_fn: Optional[ClientWeightFn] = None,
    client_weight_pow=1,
) -> tff.templates.IterativeProcess:
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    client_optimizer_fn: A function that accepts a `learning_rate` keyword
      argument and returns a `tf.keras.optimizers.Optimizer` instance.
    client_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    server_optimizer_fn: A function that accepts a `learning_rate` argument and
      returns a `tf.keras.optimizers.Optimizer` instance.
    server_lr: A scalar learning rate or a function that accepts a float
      `round_num` argument and returns a learning rate.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  client_lr_schedule = client_lr
  if not callable(client_lr_schedule):
    client_lr_schedule = lambda round_num: client_lr

  server_lr_schedule = server_lr
  if not callable(server_lr_schedule):
    server_lr_schedule = lambda round_num: server_lr

  dummy_model = model_fn()

  server_init_tf = build_server_init_fn(
      model_fn,
      # Initialize with the learning rate for round zero.
      lambda: server_optimizer_fn(server_lr_schedule(0)))
  server_state_type = server_init_tf.type_signature.result
  model_weights_type = server_state_type.model
  round_num_type = server_state_type.round_num
  elapsed_lr_type = server_state_type.elapsed_lr

  tf_dataset_type = tff.SequenceType(dummy_model.input_spec)
  model_input_type = tff.SequenceType(dummy_model.input_spec)

  @tff.tf_computation(model_input_type, model_weights_type, round_num_type, elapsed_lr_type)
  def client_update_fn(tf_dataset, initial_model_weights, round_num, elapsed_lr):
    client_lr = client_lr_schedule(round_num)
    client_optimizer = client_optimizer_fn(client_lr)
    client_update = create_client_update_fn()
    # client_update consumes two dummy model
    return client_update(model_fn(), model_fn(), tf_dataset, initial_model_weights,
                         client_optimizer, client_mirror, elapsed_lr,
                         client_weight_fn, client_weight_pow)

  @tff.tf_computation(server_state_type, model_weights_type.trainable, elapsed_lr_type)
  def server_update_fn(server_state, model_delta, elapsed_lr_delta):
    primal_model = model_fn()
    dual_model = model_fn()
    server_lr = server_lr_schedule(server_state.round_num)
    server_optimizer = server_optimizer_fn(server_lr)
    # We initialize the server optimizer variables to avoid creating them
    # within the scope of the tf.function server_update.
    _initialize_optimizer_vars(primal_model, server_optimizer)
    _initialize_optimizer_vars(dual_model, server_optimizer)
    return server_update(primal_model, dual_model, server_optimizer,
                        server_mirror, server_state, 
                        model_delta, elapsed_lr_delta)

  @tff.federated_computation(
      tff.FederatedType(server_state_type, tff.SERVER),
      tff.FederatedType(tf_dataset_type, tff.CLIENTS))
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and the result of
      `tff.learning.Model.federated_output_computation`.
    """
    client_dual_model_weights = tff.federated_broadcast(server_state.dual_model_weights)
    client_round_num = tff.federated_broadcast(server_state.round_num)
    client_elapsed_lr = tff.federated_broadcast(server_state.elapsed_lr)
    client_outputs = tff.federated_map(
        client_update_fn, 
        (federated_dataset, client_dual_model_weights,
         client_round_num, client_elapsed_lr))

    client_weight = client_outputs.client_weight
    model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_weight)

    elapsed_lr_delta = tff.federated_mean(
        client_outputs.elapsed_lr_delta, weight=client_weight)

    server_state = tff.federated_map(server_update_fn,
                   (server_state, model_delta, elapsed_lr_delta))

    aggregated_outputs = dummy_model.federated_output_computation(
        client_outputs.model_output)
    if aggregated_outputs.type_signature.is_struct():
      aggregated_outputs = tff.federated_zip(aggregated_outputs)

    return server_state, aggregated_outputs

  @tff.federated_computation
  def initialize_fn():
    return tff.federated_value(server_init_tf(), tff.SERVER)

  return tff.templates.IterativeProcess(
      initialize_fn=initialize_fn, next_fn=run_one_round)
