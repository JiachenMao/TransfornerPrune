from tensor2tensor.models import transformer
from tensor2tensor.models import basic
from tensor2tensor.utils import registry
from tensor2tensor.utils.hparam import HParams

'''
Important:
All the self-defined parameters should be first registered here
'''
def register_self_defined_params(hparams):
  
  # whether to add panelty on the word embedding layer and target space embedding during SSL training
  hparams.add_hparam("no_ssl_on_embedding", False)
  hparams.add_hparam("no_shs_on_embedding", False)

  # whether to apply regularization during training
  hparams.add_hparam("ssl_regularization", False)
  hparams.add_hparam("shs_regularization", False)

  # if apply ssl regularization loss, the decay rate of ssl
  hparams.add_hparam("ssl_decay_rate", 0.00001)
  hparams.add_hparam("shs_decay_rate", 0.001)

  # the threshold under which the weights are set as 0
  '''
  zero threshold is applied:
    (1) before every evaluation during SSL training
    (2) generate zero_mask before SSL fine tuning
  '''
  hparams.add_hparam("ssl_zero_threshold", 0.01)
  hparams.add_hparam("shs_zero_threshold", 0.01)

  # indicate whether the current training is finetuning step of ssl
  hparams.add_hparam("ssl_finetune", False)
  hparams.add_hparam("shs_finetune", False)

  # regulaization strategy during shs pruning
  hparams.add_hparam("shs_strategy", 'shs_prune_strategy_1')


  # whether to print sparsity
  hparams.add_hparam("print_sparsity", False)
  
  return hparams

@registry.register_hparams
def transformer_base_single_gpu_mjc():
  hparams = transformer.transformer_base_single_gpu()
  # hparams.eval_freq_in_steps = 100 # (2000 in default)
  # hparams.iterations_per_loop = 100 # (100 in default)
  hparams = register_self_defined_params(hparams)
  hparams.batch_size = 512
  return hparams



@registry.register_hparams
def transformer_base_single_gpu_ssl_train():
  hparams = transformer.transformer_base_single_gpu()
  hparams = register_self_defined_params(hparams)
  hparams.ssl_regularization = True
  hparams.print_sparsity = True
  hparams.ssl_finetune = False
  return hparams

@registry.register_hparams
def transformer_base_single_gpu_ssl_train_no_ssl_embedding():
  hparams = transformer_base_single_gpu_ssl_train()
  hparams.no_ssl_on_embedding = True
  return hparams


@registry.register_hparams
def transformer_base_single_gpu_shs_train():
  hparams = transformer.transformer_base_single_gpu()
  hparams = register_self_defined_params(hparams)
  hparams.shs_regularization = True
  hparams.shs_finetune = False
  hparams.print_sparsity = True
  return hparams

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding():
  hparams = transformer_base_single_gpu_shs_train()
  hparams.no_ssl_on_embedding = True
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_v2():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding()
  hparams.no_ssl_on_embedding = True
  hparams.learning_rate_constant = hparams.learning_rate_constant/10
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_v3():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding_v2()
  hparams.shs_decay_rate = 0.0001
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_v4():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding_v2()
  hparams.shs_strategy = 'shs_prune_strategy_2'
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_v5():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding_v4()
  hparams.shs_decay_rate = 0.0005
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_exp17():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding_v2()
  hparams.shs_strategy = 'shs_prune_strategy_base'
  hparams.shs_decay_rate = 0.0002
  return hparams  
@registry.register_hparams
def transformer_base_single_gpu_shs_train_no_ssl_embedding_exp18():
  hparams = transformer_base_single_gpu_shs_train_no_ssl_embedding_v2()
  hparams.shs_strategy = 'shs_prune_strategy_base'
  hparams.shs_decay_rate = 0.0005
  return hparams  

@registry.register_hparams
def transformer_base_single_gpu_ssl_finetune():
  hparams = transformer.transformer_base_single_gpu()
  hparams = register_self_defined_params(hparams)
  # duing the finetune step, the ssl regularition term will be turned off
  hparams.ssl_regularization = False
  hparams.ssl_finetune = True
  # hparams.learning_rate = hparams.learning_rate/10
  hparams.learning_rate_constant = hparams.learning_rate_constant/10
  return hparams



@registry.register_hparams
def transformer_base_single_gpu_ssl_finetune_no_ssl_embedding():
  hparams = transformer_base_single_gpu_ssl_finetune()
  hparams.no_ssl_on_embedding = True
  return hparams



'''
hparams for simple mlp on mnist with ssl training
'''
@registry.register_hparams
def mlp_mnist_ssl_train():
  hparams = basic.basic_fc_small()
  hparams = register_self_defined_params(hparams)
  # duing the train step, the ssl regularition term will be turned on
  hparams.ssl_decay_rate = 0.001
  hparams.ssl_zero_threshold = 0.01
  hparams.ssl_regularization = True
  hparams.ssl_finetune = False
  return hparams


'''
hparams for simple mlp on mnist with ssl finetuning
'''
@registry.register_hparams
def mlp_mnist_ssl_finetune():
  hparams = basic.basic_fc_small()
  hparams = register_self_defined_params(hparams)
  # duing the finetune step, the ssl regularition term will be turned off
  hparams.ssl_regularization = False
  hparams.ssl_finetune = True
  return hparams



@registry.register_pruning_params
def resnet_weight_mjc():
  hp = HParams()
  hp.add_hparam("strategy", "weight")
  hp.add_hparam("black_list", ["logits", "bias"])
  hp.add_hparam("white_list", ["conv2d"])
  hp.add_hparam("sparsities", [0.3 * i for i in range(1)])
  hp.add_hparam("weight_sparsities", [0.3 * i for i in range(1)])
  return hp


@registry.register_pruning_params
def transformer_weight_mjc():
  hp = HParams()
  hp.add_hparam("strategy", "weight")
  hp.add_hparam("black_list", ["logits", "bias"])
  hp.add_hparam("white_list", ["attention"])
  hp.add_hparam("sparsities", [0.1 * i for i in range(8)])
  hp.add_hparam("weight_sparsities", [0.1 * i for i in range(8)])
  return hp


