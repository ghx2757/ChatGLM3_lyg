#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128  # 要改此值需同步fastchat.serve.model_worker文件的parser.add_argument("--load_kwargs",type=dict,default={})default参数：
LR=2e-2
NUM_GPUS=1
MAX_SEQ_LEN=2048
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
MAX_STEP=300
SAVE_INTERVAL=100

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=data_haier_m

BASE_MODEL_PATH=D:/code/ChatGLM3_lyg/chatglm3-6b/ # 模型路径
DATASET_PATH=formatted_data/data_hsw_m.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format multi-turn \
    --train_file $DATASET_PATH \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --use_mps_device 1\
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
