#!/bin/bash
# AMCT 校准数据提取脚本
# 用法: ./extract_ptq_data.sh <model_path> <data_dir> <quant_target> <card>
# quant_target: attn-linear | mlp
# 示例: ./extract_ptq_data.sh /data/models/Qwen3.5-9B /data/ptq_data attn-linear 0
set -e

MODEL=$1
DATA_DIR=$2
TARGET=$3
CARD=${4:-0}

ASCEND_RT_VISIBLE_DEVICES=$CARD python -m amct_pytorch.extract_ptq_data \
  --model "$MODEL" --model_name qwen3_5 --data_dir "$DATA_DIR" \
  --device npu:0 --granularity block --quant_target "$TARGET" --nsamples 32
