#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL_PATH QUANT_CONFIG OUTPUT_JSON [TP_SIZE] [MEMORY_UTILIZATION]" >&2
  exit 2
fi

model_path="$1"
quant_config="$2"
output_json="$3"
tp_size="${4:-1}"
memory_utilization="${5:-0.45}"

: "${VLLM_ASCEND_ROOT:?set VLLM_ASCEND_ROOT to the vllm-ascend checkout}"
: "${VLLM_ROOT:?set VLLM_ROOT to the vLLM checkout}"

if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
  source "${CANN_ENV_SCRIPT}"
fi
if [[ -n "${VIRTUAL_ENV_ACTIVATE:-}" ]]; then
  source "${VIRTUAL_ENV_ACTIVATE}"
fi

export PYTHONPATH="${VLLM_ASCEND_ROOT}:${VLLM_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"

python "${VLLM_ASCEND_ROOT}/examples/quantization/fake_mx/validation/run_fake_mx_smoke.py" \
  "${model_path}" \
  --quant-config "${quant_config}" \
  --tensor-parallel-size "${tp_size}" \
  --gpu-memory-utilization "${memory_utilization}" \
  --output-json "${output_json}"
