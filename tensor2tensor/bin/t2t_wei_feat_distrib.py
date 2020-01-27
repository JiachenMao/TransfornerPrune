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

r"""Prune T2TModels using some heuristic.

This code demonstrates the magnitude distribution of weight and feature map.
Additionally, with the sparsity of weight increases, how accuracy drops

Target:
if the feature map (hidden state) is very sparse in nature, 
may be we can directly apply spars matrix multiplation to accelerate the model

Example run:
- train a resnet on cifar10:
    bin/t2t_trainer.py --problem=image_cifar10 --hparams_set=resnet_cifar_32 \
      --model=resnet

- evaluate different pruning percentages using weight-level pruning:
    bin/t2t_prune.py --pruning_params_set=resnet_weight --problem=image_cifar10\
      --hparams_set=resnet_cifar_32 --model=resnet
"""

import os

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem as problem_lib  # pylint: disable=unused-import
from tensor2tensor.visualization import wei_feat_distrib
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import bleu_hook

import tensorflow as tf

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("pruning_params_set", None,
                    "Which pruning parameters to use.")
flags.DEFINE_string("log_dir", '',
                    "log directory.")


# def initialize_from_ckpt(ckpt_dir):
#   tf.logging.info("Checkpoint dir: %s", ckpt_dir)
#   reader = tf.contrib.framework.load_checkpoint(ckpt_dir)
#   variable_map = {}
#   for var in tf.contrib.framework.get_trainable_variables():
#     var_name = var.name.split(":")[0]
#     if reader.has_tensor(var_name):
#       tf.logging.info("Loading variable from checkpoint: %s", var_name)
#       variable_map[var_name] = var
#     else:
#       tf.logging.info("Cannot find variable in checkpoint, skipping: %s",
#                       var_name)
#   tf.train.init_from_checkpoint(ckpt_dir, variable_map)

def create_pruning_params():
  return registry.pruning_params(FLAGS.pruning_params_set)


def create_pruning_strategy(name):
  return registry.pruning_strategy(name)


'''
get the evaluation graph for image classification
'''
def get_eval_graph_image(EstimatorSpec, labels):
  preds = EstimatorSpec.predictions["predictions"]
  preds = tf.argmax(preds, -1, output_type=labels.dtype)
  acc, acc_update_op = tf.metrics.accuracy(labels=labels, predictions=preds)
  return acc, acc_update_op

# '''
# get the evaluation graph for translation problem
# '''
# def get_eval_graph_trans(EstimatorSpec, labels):
#   preds = EstimatorSpec.predictions["predictions"]
#   # outputs = tf.to_int32(tf.argmax(preds, axis=-1))
#   outputs = tf.to_int32(tf.argmax(preds, axis=-1))
#   # Convert the outputs and labels to a [batch_size, input_length] tensor.
#   outputs = tf.squeeze(outputs, axis=[-1, -2])
#   labels = tf.squeeze(labels, axis=[-1, -2])
#   # bleu, constant = bleu_hook.bleu_score(predictions=preds, labels=labels)
#   return outputs, labels, preds

'''
get the evaluation graph for translation problem
'''
def get_eval_graph_trans(EstimatorSpec, labels):
  preds = EstimatorSpec.predictions["predictions"]
  # outputs = tf.to_int32(tf.argmax(preds, axis=-1))
  # outputs = tf.to_int32(tf.argmax(preds, axis=-1))
  # # Convert the outputs and labels to a [batch_size, input_length] tensor.
  # outputs = tf.squeeze(outputs, axis=[-1, -2])
  # labels = tf.squeeze(labels, axis=[-1, -2])
  bleu, constant = bleu_hook.bleu_score(predictions=preds, labels=labels)
  return bleu

def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()


  if FLAGS.generate_data:
    t2t_trainer.generate_data()
  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  # hparams = t2t_trainer.create_hparams()
  # hparams.add_hparam("data_dir", FLAGS.data_dir)
  # trainer_lib.add_problem_hparams(hparams, FLAGS.problem)
  hparams_path = os.path.join(FLAGS.output_dir, "hparams.json")
  hparams = trainer_lib.create_hparams(
      FLAGS.hparams_set, FLAGS.hparams, data_dir=FLAGS.data_dir,
      problem_name=FLAGS.problem, hparams_path=hparams_path)
  hparams.add_hparam("model_dir", FLAGS.output_dir)

  config = t2t_trainer.create_run_config(hparams)
  params = {"batch_size": hparams.batch_size}

  # add "_rev" as a hack to avoid image standardization
  problem = registry.problem(FLAGS.problem)
  input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                             hparams)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      FLAGS.model, hparams, use_tpu=False)

  dataset = input_fn(params, config).repeat()
  dataset_iteraor = dataset.make_one_shot_iterator()
  features, labels = dataset_iteraor.get_next()

  
  # tf.logging.info("### t2t_wei_feat_distrib.py features %s", features)
  spec = model_fn(
      features,
      labels,
      tf.estimator.ModeKeys.EVAL,
      params=hparams,
      config=config)


  # get the summary model structure graph
  summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
  
  # Restore weights
  saver = tf.train.Saver()
  checkpoint_path = os.path.expanduser(FLAGS.output_dir or
                                       FLAGS.checkpoint_path)
  tf.logging.info("### t2t_wei_feat_distrib.py checkpoint_path %s", checkpoint_path)
  # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

  # Load weights from checkpoint.
  ckpts = tf.train.get_checkpoint_state(checkpoint_path)
  ckpt = ckpts.model_checkpoint_path
  saver.restore(sess, ckpt)

  # saver.restore(sess, checkpoint_path+'/model.ckpt-1421000')
  # initialize_from_ckpt(checkpoint_path)

  # get parameter
  pruning_params = create_pruning_params()
  pruning_strategy = create_pruning_strategy(pruning_params.strategy)

  # get evalutaion graph
  if 'image' in FLAGS.problem:
    acc, acc_update_op = get_eval_graph_image(spec, labels)
    tf.summary.scalar('accuracy', acc)
    # define evaluation function
    def eval_model():
      sess.run(tf.initialize_local_variables())
      for _ in range(FLAGS.eval_steps):
        acc = sess.run(acc_update_op)
      return acc
  elif 'translate' in FLAGS.problem:
    bleu_op = get_eval_graph_trans(spec, labels)
    # tf.summary.scalar('bleu', bleu_op)
    # define evaluation function
    def eval_model():
      bleu_value = 0
      # sess.run(tf.initialize_local_variables())
      # sess.run()
      # local_vars = tf.local_variables()
      # tf.logging.info("###!!!!!!! t2t_wei_feat_distrib.py local_vars %s", local_vars)
      # for _ in range(FLAGS.eval_steps):
      for _ in range(FLAGS.eval_steps):
        # outputs_tensor, labels_tensor, preds_tensor = sess.run([outputs, labels, preds])
        bleu = sess.run(bleu_op)
        # tf.logging.info("### t2t_wei_feat_distrib.py outputs_tensor %s", outputs_tensor[0].shape)
        # tf.logging.info("### t2t_wei_feat_distrib.py labels_tensor %s", labels_tensor[0].shape)
        # tf.logging.info("### t2t_wei_feat_distrib.py preds %s", preds_tensor[0].shape)
        bleu_value += bleu
      bleu_value /= FLAGS.eval_steps
      return bleu_value

  # get weight distribution graph
  wei_feat_distrib.get_weight_distrib_graph(pruning_params)


  
  # do accuracy sparsity tradeoff for model weights
  wei_feat_distrib.wei_sparsity_acc_tradeoff(sess, eval_model, pruning_strategy, pruning_params, summary_writer)

  # do accuracy sparsity tradeoff for model weights

  # save the summary
  summary_writer.close()


  sess.run(tf.initialize_local_variables())
  preds = spec.predictions["predictions"]
  # features_shape=tf.shape(features)
  pred_shape=tf.shape(preds)
  labels_shape=tf.shape(labels)
  # tf.logging.info("###---- t2t_wei_feat_distrib.py feature preds %s", features)
  # tf.logging.info("###---- t2t_wei_feat_distrib.py shape preds %s", sess.run([pred_shape, labels_shape]))
  # tf.logging.info("###---- t2t_wei_feat_distrib.py shape labels_shape %s", sess.run(labels_shape))

  # print weight distribution to terminal
  # wei_feat_distrib.print_weight_distrib(sess, pruning_params)





if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()










  
