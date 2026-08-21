#!/bin/bash
# 通用 vllm serve 启动脚本
# 用法: ./vllm_serve.sh <model_path> <card_id> <port> <config_json> [--enforce-eager]
# 示例: ./vllm_serve.sh /data/models/Qwen3.5-9B 0 8001 configs/qwen3_5_9b_rtn_attn-only_w4a4.json
set -e

MODEL_PATH=$1
CARD=$2
PORT=$3
CONFIG=$4
EAGER=${5:-}

export PYTHONPATH=$(pwd)/../../../../../:${PYTHONPATH:-}

cp "$CONFIG" "$MODEL_PATH/quant_model_description.json"

if [ "$EAGER" = "--enforce-eager" ]; then
  ASCEND_RT_VISIBLE_DEVICES=$CARD vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 1 --served-model-name qwen3.5 \
    --max-num-seqs 32 --max-model-len 16384 \
    --trust-remote-code --gpu-memory-utilization 0.90 \
    --mamba-ssm-cache-dtype bfloat16 --dtype bfloat16 \
    --quantization ascend --enforce-eager
else
  ASCEND_RT_VISIBLE_DEVICES=$CARD vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 1 --served-model-name qwen3.5 \
    --max-num-seqs 32 --max-model-len 16384 \
    --trust-remote-code --gpu-memory-utilization 0.90 \
    --mamba-ssm-cache-dtype bfloat16 --dtype bfloat16 \
    --quantization ascend \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16]}'
fi
