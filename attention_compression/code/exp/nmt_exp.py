import os
current_dir = os.path.dirname(os.path.realpath(__file__))
import sys
# add the tensor2tensor lib path
sys.path.append(os.path.abspath(current_dir+'/../../../'))
# add the compression component lib path
sys.path.append(os.path.abspath(current_dir+'/../'))

import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import collections
import yaml
import argparse

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics


from config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Attention model compression')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))

# Restore and translate!
def translate(inputs, tfe, ckpt_path, translate_model):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create(ckpt_path):
    model_output = translate_model.infer(encoded_inputs)["outputs"]
  return decode(model_output)

def train():
  tf.logging.set_verbosity(tf.logging.INFO)

  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # If we just have to print the registry, do that and exit early.
  maybe_log_registry_and_exit()

  # Create HParams.
  if argv:
    set_hparams_from_args(argv[1:])
  hparams = create_hparams()

  if FLAGS.schedule == "train" or FLAGS.schedule == "train_eval_and_decode":
    mlperf_log.transformer_print(key=mlperf_log.RUN_START, hparams=hparams)
  if FLAGS.schedule == "run_std_server":
    run_std_server()
  mlperf_log.transformer_print(
      key=mlperf_log.RUN_SET_RANDOM_SEED, value=FLAGS.random_seed,
      hparams=hparams)
  trainer_lib.set_random_seed(FLAGS.random_seed)

  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  if FLAGS.generate_data:
    generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  exp_fn = create_experiment_fn()
  exp = exp_fn(create_run_config(hparams), hparams)
  if is_chief():
    save_metadata(hparams)
  execute_schedule(exp)
  if FLAGS.schedule != "train":
    mlperf_log.transformer_print(key=mlperf_log.RUN_FINAL,
                                 hparams=hparams)


parse_args()

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# setup GPU
os.environ["CUDA_VISIBLE_DEVICES"]=config.GPU

gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"

# # Setup some directories
data_dir = os.path.abspath(current_dir+config.DATA_PATH)
tmp_dir = os.path.abspath(current_dir+config.TMP_PATH)
checkpoint_dir = os.path.abspath(current_dir+config.MODEL_PATH)
log_dir = os.path.abspath(current_dir+config.LOG_PATH)

tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(checkpoint_dir)
tf.gfile.MakeDirs(log_dir)

# A Problem is a dataset together with some fixed pre-processing.
# It could be a translation dataset with a specific tokenization,
# or an image dataset with a specific resolution.
#
# There are many problems available in Tensor2Tensor
# problems.available()

# Fetch the problem
problem = problems.problem(config.PROBLEM)

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_name = config.VOCAB_NAME
vocab_file = os.path.join(gs_data_dir, vocab_name)
cp_cmd = 'gsutil cp '+vocab_file+' ' + data_dir
os.system(cp_cmd)
# !gsutil cp {vocab_file} {data_dir}

# Get the encoders from the problem
encoders = problem.feature_encoders(data_dir)


# download the dataset for training and testing
problem.generate_data(data_dir, tmp_dir)

if not config.TEST_ONLY:
  example = tfe.Iterator(ende_problem.dataset(Modes.TRAIN, data_dir)).next()                                                                                                                                                           problem.dataset(Modes.TRAIN, data_dir)).next()
  inputs = [int(x) for x in example["inputs"].numpy()] # Cast to ints.
  targets = [int(x) for x in example["targets"].numpy()] # Cast to ints.


  # Create hparams and the model
  model_name = config.MODEL_NAME
  hparams_set = config.HPARAMS_SET

  hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name=config.PROBLEM)

  # NOTE: Only create the model once when restoring from a checkpoint; it's a
  # Layer and so subsequent instantiations will have different variable scopes
  # that will not match the checkpoint.
  translate_model = registry.model(model_name)(hparams, Modes.EVAL)
  if config.TRAIN.finetune:
    # Copy the pretrained checkpoint locally
    ckpt_name = config.TRAIN.pretrained_model_name
    gs_ckpt = os.path.join(gs_ckpt_dir, ckpt_name)
    pretrained_dir = os.path.abspath(current_dir+config.TRAIN.pretrained_model_path)
    tf.gfile.MakeDirs(pretrained_dir)
    local_ckpt_path = os.path.join(pretrained_dir, ckpt_name)
    # if not exist locally, download the pretrained model
    if not os.path.isfile(local_ckpt_path):
      cp_cmd = 'gsutil -q cp -R '+gs_ckpt+' ' + pretrained_dir
      os.system(cp_cmd)
    ckpt_path = tf.train.latest_checkpoint(local_ckpt_path)



    inputs = "The animal didn't cross the street because it was too tired"
    outputs = translate(inputs, tfe, ckpt_path, translate_model)

    print("Inputs: %s" % inputs)
    print("Outputs: %s" % outputs)


