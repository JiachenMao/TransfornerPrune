# demo code to train mnist with a simple model structure
# source activate t2t
python ./code/exp/mnist_exp.py --cfg code/exp/cfg/mnist_demo.yaml


# purely fine-tune (no ssl or low rank) transformer model from a pre-trained model
# source activate t2t
python ./code/exp/nmt_exp.py --cfg code/exp/cfg/transformer/nmt_eng2ger_finetune.yaml