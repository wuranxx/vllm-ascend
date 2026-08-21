#!/bin/bash
# AMCT PTQ 多卡并行训练脚本
# 用法: ./run_amct_ptq.sh <algo> <quant_target> <model_path> <data_dir> <output_dir>
# algo:         flatquant | learnable_had
# quant_target: attn-linear | mlp
# 示例: ./run_amct_ptq.sh flatquant attn-linear /data/models/Qwen3.5-9B /data/ptq_data /data/ptq_out
set -e

ALGO=$1
TARGET=$2
MODEL=$3
DATA_DIR=$4
OUTPUT_DIR=$5
BIT_CONFIG=${6:-amct/amct_pytorch/configs/w4a4_mxfp.yaml}

NUM_BLOCKS=32
NUM_TASKS=${NPU_TASKS:-8}
CARDS_STR=${NPU_CARDS:-"0 1 2 3 4 5 6 7"}

avg=$((NUM_BLOCKS / NUM_TASKS))
rem=$((NUM_BLOCKS % NUM_TASKS))
start=0
read -ra CARDS <<< "$CARDS_STR"

for ((i=0; i<NUM_TASKS; i++)); do
  [ $i -lt $rem ] && len=$((avg+1)) || len=$avg
  end=$((start+len))
  npu=${CARDS[$i]}
  echo "Task $i: blocks [$start,$end) npu=$npu"
  ASCEND_RT_VISIBLE_DEVICES=$npu python -m amct_pytorch.ptq \
    --model "$MODEL" --model_name qwen3_5 --data_dir "$DATA_DIR" \
    --device npu:0 --granularity block \
    --start_block_idx $start --end_block_idx $end \
    --quant_target "$TARGET" --quant_dtype mxfp --bit_config "$BIT_CONFIG" \
    --algos "$ALGO" --output_dir "$OUTPUT_DIR" \
    --epochs 15 --base_lr 1e-5 --cali_bsz 2 --nsamples 32 &
  start=$end
  [ $i -lt $((NUM_TASKS-1)) ] && sleep 15
done
wait
echo "All PTQ tasks done."
