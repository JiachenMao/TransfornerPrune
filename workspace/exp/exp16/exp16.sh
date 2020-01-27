
######### the same as exp 15 except change  shs_decay_rate from 0.001 to 0.0005 ########

cd /home/jm562/Desktop/project5/tensor2tensor/workspace/
source activate t2t



LOCAL_EVAL_FREQUENCY=2000
TRAIN_STEPS=2000
EVAL_STEPS=50
# "train", "train_and_evaluate", "continuous_train_and_eval"
SCHEDULE=continuous_train_and_eval # continuous_train_and_eval in default
MAX_NUM_CKPT=1
EVAL_INTERVAL=30  # evaluate every N seconds
PROJECT_ROOT_DIR=/home/jm562/Desktop/project5/tensor2tensor/workspace

EXP_NAME=exp16

PROBLEM=translate_ende_wmt32k

MODEL=transformer # model structure of the original model


# HPARAMS=basic_fc_small
HPARAMS=transformer_base_single_gpu_shs_train_no_ssl_embedding_v5
# HPARAMS=resnet_cifar_32



DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
LOG_DIR=$TRAIN_DIR/log

PRETRAINED_MODEL=transformer_ende_test



ORI_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ori_pretrained_model
SHS_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/shs_pretrained_model

rm -r $TRAIN_DIR
rm -r $LOG_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $LOG_DIR $ORI_PRE_TRAIN_MODEL_DIR $SHS_PRE_TRAIN_MODEL_DIR


# download pretrained mdoel to output dir
gs_ckpt_dir=gs://tensor2tensor-checkpoints/
gs_ckpt=$gs_ckpt_dir$PRETRAINED_MODEL
gsutil -q cp -R $gs_ckpt $ORI_PRE_TRAIN_MODEL_DIR

t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM


################### Do shs training ##################
WARM_START_DIR=$ORI_PRE_TRAIN_MODEL_DIR
TRAIN_STEPS=200000
rm -r $SHS_PRE_TRAIN_MODEL_DIR # before shs finetuning, make sure the output dir is empty
mkdir -p $SHS_PRE_TRAIN_MODEL_DIR

rm $LOG_DIR/output.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID 
export CUDA_VISIBLE_DEVICES=2
nohup t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$SHS_PRE_TRAIN_MODEL_DIR \
  --warm_start_from=$WARM_START_DIR/$PRETRAINED_MODEL \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS >> $LOG_DIR/output.log 2>&1 &

cat $LOG_DIR/output.log
