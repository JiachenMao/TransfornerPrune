# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to assist in pruning models."""

import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry

import tensorflow as tf
import numpy as np
import math

@registry.register_pruning_strategy
def weight(w, sparsity):
  """Weight-level magnitude pruning."""
  w_shape = common_layers.shape_list(w)
  k = int(np.prod(w_shape[:-1]))
  count = tf.to_int32(k * sparsity)
  mask = common_layers.weight_targeting(w, count)
  return (1 - mask) * w if len(w_shape)>1 else w


@registry.register_pruning_strategy
def unit(w, sparsity):
  """Unit-level magnitude pruning."""
  w_shape = common_layers.shape_list(w)
  count = tf.to_int32(w_shape[-1] * sparsity)
  mask = common_layers.unit_targeting(w, count)
  return (1 - mask) * w if len(w_shape)>1 else w

def should_show(name, pruning_params):
    """Whether to show the layer's distribution or not."""
    in_whitelist = not pruning_params.white_list or any(
        e in name for e in pruning_params.white_list)
    in_blacklist = any(e in name for e in pruning_params.black_list)
    if pruning_params.white_list and not in_whitelist:
      return False
    elif in_blacklist:
      return False
    return True

def feat_sparsity_acc_tradeoff():
  pass

# def wei_sparsity_acc_tradeoff():
#   pass

def feat_distrib(sess, eval_model, pruning_strategy, pruning_params, num_steps=10):
  pass

def get_weight_distrib_graph(pruning_params):
  weights = tf.trainable_variables()
  weights = [w for w in weights if should_show(w.name, pruning_params)]
  weights_name = [w.name for w in weights]
  for weight, weight_name in zip(weights, weights_name):
    tf.summary.histogram('Weight Hist-'+weight_name, weight)


def print_weight_distrib(sess, pruning_params, num_steps=10):
  weights = tf.trainable_variables()
  # tf.logging.info("### wei_feat_distrib.py  layer names %s", weights)
  weights = [w for w in weights if should_show(w.name, pruning_params)]
  weights_name = [w.name for w in weights]
  weights = sess.run(weights)
  for weight, weight_name in zip(weights, weights_name):
    weight = np.absolute(weight)
    max_weight = weight.max()
    hist, bin_edges = np.histogram(weight, 
      bins=np.arange(0, max_weight+0.000001, max_weight/num_steps), 
      density=True)
    hist /= hist.sum()
    tf.logging.info("\n ---"+weight_name+"\n Hist: %s  \n Range: (%0.1f - %0.5f) \n Step: %0.5f" , 
      hist, bin_edges[0], bin_edges[-1], max_weight/num_steps)
    


def wei_sparsity_acc_tradeoff(sess, eval_model, pruning_strategy, pruning_params, summary_writer):
  """Prune the weights of a model and evaluate."""
  # tf.logging.info("### wei_feat_distrib.py Weight sparsity accuracy tradeoff ###")
  weights = tf.trainable_variables()
  weights = [w for w in weights if should_show(w.name, pruning_params)]
  # tf.logging.info("Pruning weights: %s" % weights.shape)
  # tf.logging.info("Pruning weights: %s" % weights[1])
  unpruned_weights = sess.run(weights)
  # tf.logging.info("debugggg: %s" % unpruned_weights[1])

  reset_op = tf.no_op()
  for w, ow in zip(weights, unpruned_weights):
    op = tf.assign(w, ow)
    reset_op = tf.group(reset_op, op)

  for step, sparsity in enumerate(pruning_params.weight_sparsities):
    set_weights_op = tf.no_op()
    for w in weights:
      op = tf.assign(w, pruning_strategy(w, sparsity))
      set_weights_op = tf.group(set_weights_op, op)
    sess.run(set_weights_op)

    acc = eval_model()
    tf.logging.info("\tPruning to sparsity = %f: acc or bleu = %f" % (sparsity, acc))
    sess.run(reset_op)
    merged = tf.summary.merge_all()
    summary = sess.run(merged)
    summary_writer.add_summary(summary, step)








