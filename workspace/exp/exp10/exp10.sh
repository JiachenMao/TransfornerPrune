######### transformer ########
# in this exp 10 we just want to do ssl finetune to transformer like exp7
# except that embedding layer and target space embedding are not applied with SSL


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

EXP_NAME=exp10

PROBLEM=translate_ende_wmt32k

MODEL=transformer # model structure of the original model

SSL_MODEL=own_transformer_ssl # model structure of the model during ssl training

LOW_RANK_MODEL=simple_mlp_low_rank_1 # model structure of the model during low_rank、 training


# HPARAMS=basic_fc_small
HPARAMS=transformer_base_single_gpu_ssl_train_no_ssl_embedding
# HPARAMS=resnet_cifar_32



DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS
LOG_DIR=$TRAIN_DIR/log

PRETRAINED_MODEL=transformer_ende_test



ORI_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ori_pretrained_model
SSL_PRE_TRAIN_MODEL_DIR=$TRAIN_DIR/ssl_pretrained_model

rm -r $TRAIN_DIR
rm -r $LOG_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $LOG_DIR $ORI_PRE_TRAIN_MODEL_DIR $SSL_PRE_TRAIN_MODEL_DIR


# download pretrained mdoel to output dir
gs_ckpt_dir=gs://tensor2tensor-checkpoints/
gs_ckpt=$gs_ckpt_dir$PRETRAINED_MODEL
gsutil -q cp -R $gs_ckpt $ORI_PRE_TRAIN_MODEL_DIR

t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM


################### Do ssl training ##################
WARM_START_DIR=$ORI_PRE_TRAIN_MODEL_DIR
TRAIN_STEPS=20000
# HPARAMS=mlp_mnist_ssl_train
rm -r $SSL_PRE_TRAIN_MODEL_DIR # before ssl finetuning, make sure the output dir is empty
mkdir -p $SSL_PRE_TRAIN_MODEL_DIR

rm $LOG_DIR/output.log
nohup t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$SSL_MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$SSL_PRE_TRAIN_MODEL_DIR \
  --warm_start_from=$WARM_START_DIR/$PRETRAINED_MODEL \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY \
  --train_steps=$TRAIN_STEPS \
  --keep_checkpoint_max=$MAX_NUM_CKPT \
  --schedule=$SCHEDULE \
  --eval_throttle_seconds=$EVAL_INTERVAL \
  --eval_steps=$EVAL_STEPS >> $LOG_DIR/output.log 2>&1 &

cat $LOG_DIR/output.log




################ Get BLEU score of origianl model on newstest2014#################
TEST_DATA_DIR=$PROJECT_ROOT_DIR/test_data
BEAM_SIZE=4
ALPHA=0.6
DECODE_FILE=$TEST_DATA_DIR/newstest2014.en
REF_TRANS=$TEST_DATA_DIR/newstest2014.de
TRANS=$ORI_PRE_TRAIN_MODEL_DIR/transformer_ende_test/translation_newstest2014.en

#DECODE_FILE=$ORI_PRE_TRAIN_MODEL_DIR/transformer_ende_test/newstest2012.en
#REF_TRANS=$ORI_PRE_TRAIN_MODEL_DIR/transformer_ende_test/newstest2014.tok.de
#TRANS=$ORI_PRE_TRAIN_MODEL_DIR/transformer_ende_test/newstest2014.en.transformer.transformer_base.translate_ende_wmt32k.beam4.alpha0.6.decodes
#echo "Hello world" >> $DECODE_FILE
#echo "Goodbye world" >> $DECODE_FILE
#echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de
rm $TRANS
t2t-decoder \
 --data_dir=$DATA_DIR \
 --problem=$PROBLEM \
 --model=$MODEL \
 --hparams_set=$HPARAMS \
 --output_dir=$ORI_PRE_TRAIN_MODEL_DIR/$PRETRAINED_MODEL \
 --t2t_usr_dir=$T2T_USER_DIR \
 --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
 --decode_from_file=$DECODE_FILE \
 --decode_to_file=$TRANS

# See the translations
#cat $DECODE_FILE

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=$TRANS --reference=$REF_TRANS
# BLEU_uncased =  26.92
# BLEU_cased =  26.40
