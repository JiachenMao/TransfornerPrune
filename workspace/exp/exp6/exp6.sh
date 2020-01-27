######### mnist mlp ########
# Previously, we could fine tune from a pretrained model with 
# the same model structure. However, in our situation, we need 
# to port a pretrained weight matrix to several weight matrix, 
# which means the model structure is also changed. Therefore, 
# we need to realize the feature which could save a pretrained model 
# layer and automatically initialize several layers in new model.



cd /home/jm562/Desktop/project5/tensor2tensor/workspace/
source activate t2t



LOCAL_EVAL_FREQUENCY=1000
TRAIN_STEPS=1000
EVAL_STEPS=50
# "train", "train_and_evaluate", "continuous_train_and_eval"
SCHEDULE=continuous_train_and_eval # continuous_train_and_eval in default
MAX_NUM_CKPT=1
EVAL_INTERVAL=30  # evaluate every N seconds
PROJECT_ROOT_DIR=/home/jm562/Desktop/project5/tensor2tensor/workspace

EXP_NAME=exp6

PROBLEM=image_mnist

MODEL=simple_mlp # model structure of the original model

SSL_MODEL=simple_mlp_ssl_1 # model structure of the model during ssl training

LOW_RANK_MODEL=simple_mlp_low_rank_1 # model structure of the model during low_rank、 training


HPARAMS=basic_fc_small
# HPARAMS=resnet_cifar_32



DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
LOG_DIR=$TRAIN_DIR/log


ORI_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ori_pretrained_model
SSL_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ssl_pretrained_model

rm -r $TRAIN_DIR
rm -r $LOG_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $LOG_DIR $ORI_PRE_TRAIN_MODEL_DIR $SSL_PRE_TRAIN_MODEL_DIR

t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

################### Do normal training ##################
rm -r $ORI_PRE_TRAIN_MODEL_DIR # before ssl finetuning, make sure the output dir is empty
mkdir -p $ORI_PRE_TRAIN_MODEL_DIR
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$ORI_PRE_TRAIN_MODEL_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS

################### Do ssl training ##################
WARM_START_DIR=$ORI_PRE_TRAIN_MODEL_DIR
TRAIN_STEPS=2000
HPARAMS=mlp_mnist_ssl_train
rm -r $SSL_PRE_TRAIN_MODEL_DIR # before ssl finetuning, make sure the output dir is empty
mkdir -p $SSL_PRE_TRAIN_MODEL_DIR
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$SSL_MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$SSL_PRE_TRAIN_MODEL_DIR \
  --warm_start_from=$WARM_START_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS



# PRUNING_PARAMS_SET=transformer_weight_mjc

# PRETRAINED_MODEL=transformer_ende_test



# # rm -r $TRAIN_DIR
# rm -r $LOG_DIR
# mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $LOG_DIR


# t2t-datagen \
#   --data_dir=$DATA_DIR \
#   --tmp_dir=$TMP_DIR \
#   --problem=$PROBLEM

# # download pretrained mdoel to output dir
# gs_ckpt_dir=gs://tensor2tensor-checkpoints/
# gs_ckpt=$gs_ckpt_dir$PRETRAINED_MODEL
# gsutil -q cp -R $gs_ckpt $TRAIN_DIR



# ################### Do normal training ##################
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
#   --data_dir=$DATA_DIR \
#   --problem=$PROBLEM \
#   --model=$MODEL \
#   --hparams_set=$HPARAMS \
#   --output_dir=$TRAIN_DIR \
#   --t2t_usr_dir=$T2T_USER_DIR \
#   --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
#   --train_steps=$TRAIN_STEPS \
#   --hparams="batch_size=$BATCH_SIZE" \
#   --hparams="learning_rate=$LEARNING_RATE" \
#   --keep_checkpoint_max=$MAX_NUM_CKPT \
#   --schedule=$SCHEDULE \
#   --eval_throttle_seconds=$EVAL_INTERVAL \
#   --eval_steps=$EVAL_STEPS


# ################### Do ssl training ##################
# WARM_START_DIR=$TRAIN_DIR
# TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
# SSL_REGULARIZATION_DECAY=0.001
# SSL_ZERO_THRESHOLD=0.01
# TRAIN_STEPS=1431000
# # rm -r $TRAIN_DIR
# # mkdir -p $TRAIN_DIR
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
#   --data_dir=$DATA_DIR \
#   --problem=$PROBLEM \
#   --model=$MODEL \
#   --hparams_set=$HPARAMS \
#   --output_dir=$TRAIN_DIR \
#   --warm_start_from=$WARM_START_DIR \
#   --t2t_usr_dir=$T2T_USER_DIR \
#   --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
#   --train_steps=$TRAIN_STEPS \
#   --hparams="batch_size=$BATCH_SIZE" \
#   --hparams="learning_rate=$LEARNING_RATE" \
#   --hparams="ssl_decay_rate=$SSL_REGULARIZATION_DECAY" \
#   --hparams="ssl_zero_threshold=$SSL_ZERO_THRESHOLD" \
#   --keep_checkpoint_max=$MAX_NUM_CKPT \
#   --schedule=$SCHEDULE \
#   --eval_throttle_seconds=$EVAL_INTERVAL \
#   --eval_steps=$EVAL_STEPS

