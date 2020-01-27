
# 1000 in default (save_ckpt_steps = max(FLAGS.iterations_per_loop, 
# FLAGS.local_eval_frequency))

cd /home/jm562/Desktop/project5/tensor2tensor/workspace/
source activate t2t

LOCAL_EVAL_FREQUENCY=100

EXP_NAME=exp1

PROBLEM=translate_ende_wmt32k

MODEL=transformer

PRETRAINED_MODEL=transformer_ende_test

# HPARAMS=transformer_base_single_gpu
# HPARAMS=transformer_base_single_gpu_ssl_train
HPARAMS=transformer_base_single_gpu_ssl_finetune


PROJECT_ROOT_DIR=/home/jm562/Desktop/project5/tensor2tensor/workspace
# t2t-trainer --registry_help

DATA_DIR=$PROJECT_ROOT_DIR/data
TMP_DIR=$PROJECT_ROOT_DIR/tmp
PRETRAINED_MODEL_DIR=$PROJECT_ROOT_DIR/pretrained_model
T2T_USER_DIR=$PROJECT_ROOT_DIR/own_component
TRAIN_DIR=$PROJECT_ROOT_DIR/output/$EXP_NAME/$PROBLEM/$MODEL-$HPARAMS

# rm -r $TRAIN_DIR
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $PRETRAINED_MODEL_DIR

# Generate data
# t2t-datagen \
#   --data_dir=$DATA_DIR \
#   --tmp_dir=$TMP_DIR \
#   --problem=$PROBLEM


# prepare pretrained model
gs_ckpt_dir=gs://tensor2tensor-checkpoints/
gs_ckpt=$gs_ckpt_dir$PRETRAINED_MODEL

# download pretrained mdoel
# the following command only need to be executed once
gsutil -q cp -R $gs_ckpt $PRETRAINED_MODEL_DIR


# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
# t2t-trainer \
#   --data_dir=$DATA_DIR \
#   --problem=$PROBLEM \
#   --model=$MODEL \
#   --hparams_set=$HPARAMS \
#   --output_dir=$TRAIN_DIR \
#   --hparams='batch_size=1024' \
#   --t2t_usr_dir=$T2T_USER_DIR

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=1024' \
  --warm_start_from=$PRETRAINED_MODEL_DIR/$PRETRAINED_MODEL \
  --t2t_usr_dir=$T2T_USER_DIR \
  --local_eval_frequency=$LOCAL_EVAL_FREQUENCY # (1000 in default)


# Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# See the translations
cat translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.en --reference=ref-translation.de


