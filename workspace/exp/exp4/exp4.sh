
######### on mnist mlp ########
# go through (1) SSL training and (2) SSL fintuning so as to verify the correctness of our SSL code


cd /home/jm562/Desktop/project5/tensor2tensor/workspace/
source activate t2t

LOCAL_EVAL_FREQUENCY=1000

TRAIN_STEPS=10000

EVAL_STEPS=50

EXP_NAME=exp4

PROBLEM=image_mnist


MODEL=simple_mlp
# MODEL=resnet

# HPARAMS=basic_fc_small
HPARAMS=resnet_cifar_32

# "train", "train_and_evaluate", "continuous_train_and_eval"
SCHEDULE=continuous_train_and_eval # continuous_train_and_eval in default

BATCH_SIZE=256

LEARNING_RATE=0.001

MAX_NUM_CKPT=10

EVAL_INTERVAL=30  # evaluate every N seconds

PROJECT_ROOT_DIR=/home/jm562/Desktop/project5/tensor2tensor/workspace
# t2t-trainer --registry_help

DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS


rm -r $TRAIN_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR



################### Generate data ##################
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM



################### Do normal training ##################
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --hparams="batch_size=$BATCH_SIZE" \
  --hparams="learning_rate=$LEARNING_RATE" \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS



################### Do ssl training ##################
HPARAMS=mlp_mnist_ssl_train
WARM_START_DIR=$TRAIN_DIR
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
SSL_REGULARIZATION_DECAY=0.1
SSL_ZERO_THRESHOLD=0.01
TRAIN_STEPS=20000
# rm -r $TRAIN_DIR
# mkdir -p $TRAIN_DIR
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --warm_start_from=$WARM_START_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --hparams="batch_size=$BATCH_SIZE" \
  --hparams="learning_rate=$LEARNING_RATE" \
  --hparams="ssl_decay_rate=$SSL_REGULARIZATION_DECAY" \
  --hparams="ssl_zero_threshold=$SSL_ZERO_THRESHOLD" \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS


################### Do ssl fine-tuning ##################
HPARAMS=mlp_mnist_ssl_finetune
WARM_START_DIR=$TRAIN_DIR
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
TRAIN_STEPS=30000
mkdir -p $TRAIN_DIR
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --warm_start_from=$WARM_START_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --hparams="batch_size=$BATCH_SIZE" \
  --hparams="learning_rate=$LEARNING_RATE" \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS

























