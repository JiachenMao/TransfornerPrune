######### transformer ########
# in exp7, we train the transformer with ssl train
# In this exp8, we try to rescue the accuracy back by ssl fintuning



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

EXP_NAME=exp8

PROBLEM=translate_ende_wmt32k

MODEL=transformer # model structure of the original model

SSL_MODEL=own_transformer_ssl # model structure of the model during ssl training

LOW_RANK_MODEL=simple_mlp_low_rank_1 # model structure of the model during low_rank、 training



HPARAMS=transformer_base_single_gpu_ssl_finetune


DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
LOG_DIR=$TRAIN_DIR/log



SSL_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ssl_pretrained_model
SSL_FINE_TUNED_MODEL_DIR=$TRAIN_DIR/ssl_finetuned_model

rm -r $TRAIN_DIR
rm -r $LOG_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $LOG_DIR  $SSL_PRE_TRAIN_MODEL_DIR $SSL_FINE_TUNED_MODEL_DIR


# copy ssl training model from exp7 to exp8
cp ~/Desktop/project5/tensor2tensor/workspace/output/exp7/translate_ende_wmt32k/transformer-transformer_base_single_gpu_mjc/ssl_pretrained_model/* $SSL_PRE_TRAIN_MODEL_DIR


t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM


################### Do ssl finetuning ##################
WARM_START_DIR=$SSL_PRE_TRAIN_MODEL_DIR
TRAIN_STEPS=160000
rm -r $SSL_FINE_TUNED_MODEL_DIR # before ssl finetuning, make sure the output dir is empty
mkdir -p $SSL_FINE_TUNED_MODEL_DIR


rm $LOG_DIR/output.log
nohup t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$SSL_MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$SSL_FINE_TUNED_MODEL_DIR \
  --warm_start_from=$WARM_START_DIR \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS >> $LOG_DIR/output.log 2>&1 &

cat $LOG_DIR/output.log


