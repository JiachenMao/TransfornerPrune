'''
define all the self-designed model structure here
'''

import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from own_layer import dense_block_wise_decouple

@registry.register_model
class simple_conv(t2t_model.T2TModel):
  def body(self, features):
    inputs = features["inputs"]
    filters = 10
    h1 = tf.layers.conv2d(inputs, filters,
                          kernel_size=(5, 5), strides=(2, 2))
    h2 = tf.layers.conv2d(tf.nn.relu(h1), filters,
                          kernel_size=(5, 5), strides=(2, 2))
    return tf.layers.conv2d(tf.nn.relu(h2), filters,
                            kernel_size=(3, 3))



@registry.register_model
class simple_mlp(t2t_model.T2TModel):
  def body(self, features):
    hp = self.hparams
    unit = 64
    num_classes = self._problem_hparams.vocab_size["targets"]
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    if is_training:
      targets = features["targets_raw"]
    inputs = features["inputs"]
    # flatten_input = tf.layers.flatten(inputs)
    shape = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [-1, shape[1] * shape[2] * shape[3]])
    # with tf.variable_scope("dense_1"):
    h1 = tf.layers.dense(inputs, unit, name='dense_1')
    # with tf.variable_scope("dense_2"):
    h2 = tf.layers.dense(tf.nn.relu(h1), unit, name='dense_2')
    # with tf.variable_scope("dense_3"):
    logits = tf.layers.dense(tf.nn.relu(h2), num_classes, name='dense_3')
    losses = {"training": 0.0}
    if is_training:
      tf.logging.info(" is training: num_classes %d" % num_classes)
      tf.logging.info(" is training: num_classes %s" % logits.shape)
      loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.squeeze(targets), logits=logits)
      loss = tf.reduce_mean(loss)
      losses = {"training": loss}
    logits = tf.reshape(logits, [-1, 1, 1, 1, logits.shape[1]])
    return logits, losses


@registry.register_model
class simple_mlp_ssl_1(t2t_model.T2TModel):
  def body(self, features):
    hp = self.hparams
    unit = 64
    num_classes = self._problem_hparams.vocab_size["targets"]
    is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
    if is_training:
      targets = features["targets_raw"]
    inputs = features["inputs"]
    # flatten_input = tf.layers.flatten(inputs)
    shape = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [-1, shape[1] * shape[2] * shape[3]])
    h1 = tf.layers.dense(inputs, unit, name='dense_1')
    h2 = dense_block_wise_decouple(tf.nn.relu(h1), unit, in_group=2, out_group=2, name='dense_2', use_bias=True)
    # h2 = tf.layers.dense(tf.nn.relu(h1), unit)
    logits = tf.layers.dense(tf.nn.relu(h2), num_classes, name='dense_3')

    losses = {"training": 0.0}
    if is_training:
      tf.logging.info(" is training: num_classes %d" % num_classes)
      tf.logging.info(" is training: num_classes %s" % logits.shape)
      loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.squeeze(targets), logits=logits)
      loss = tf.reduce_mean(loss)
      losses = {"training": loss}
    logits = tf.reshape(logits, [-1, 1, 1, 1, logits.shape[1]])
    return logits, losses




































