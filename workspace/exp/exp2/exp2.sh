
######### on cifar 10 resnet ########
# (1) visualize weight , feature map magnitude distribution
# (2) how accuracy drops with different weight, feature map sparsity
# (3) structure visualize on tensorboard

cd /home/jm562/Desktop/project5/tensor2tensor/workspace/
source activate t2t

PROJECT_ROOT_DIR=/home/jm562/Desktop/project5/tensor2tensor/workspace
EXP_NAME=exp2
# PROBLEM=translate_ende_wmt32k
PROBLEM=image_cifar10
# HPARAMS=transformer_base_single_gpu_mjc
HPARAMS=resnet_cifar_32
# MODEL=transformer
MODEL=resnet
PRUNING_PARAMS_SET=resnet_weight_mjc

PRETRAINED_MODEL=transformer_ende_test


DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
PRETRAINED_MODEL_DIR=$PROJECT_ROOT_DIR/pretrained_model
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
LOG_DIR=$TRAIN_DIR/log

# rm -r $TRAIN_DIR
rm -r $LOG_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $PRETRAINED_MODEL_DIR $LOG_DIR


t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

t2t-trainer --problem=$PROBLEM \
      --data_dir=$DATA_DIR \
      --hparams_set=$HPARAMS \
      --model=$MODEL \
      --output_dir=$TRAIN_DIR \
      --t2t_usr_dir=$T2T_USER_DIR


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 t2t-wei_feat_distrib \
      --pruning_params_set=$PRUNING_PARAMS_SET \
      --problem=$PROBLEM\
      --hparams_set=$HPARAMS \
      --model=$MODEL \
      --output_dir=$TRAIN_DIR \
      --t2t_usr_dir=$T2T_USER_DIR \
      --data_dir=$DATA_DIR \
      --log_dir=$LOG_DIR


tensorboard --logdir=$LOG_DIR

# access http://localhost:6006 to view
# For other device:
# access http://10.236.176.28:6006/#scalars

