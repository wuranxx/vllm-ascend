#!/usr/bin/env python3
"""MMLU-Pro 评测脚本（5-shot MCQ，12032题，无需 sandbox）。

用法: python run_mmlu_pro.py <port> <work_dir>
示例: python run_mmlu_pro.py 8001 ./outputs/omniquant_attn-only_mmlu_pro

环境要求:
  conda activate xyj  (evalscope 1.9.1, pyarrow 19.0.1)
  vllm serve 需在 vllm-qwen35-9b-x 容器中以 host 网络模式启动

数据集缓存:
  MODELSCOPE_CACHE=/data2/x00823151/datasets (已下载)
"""

import os
import sys

os.environ.setdefault("MODELSCOPE_CACHE", "/data2/x00823151/datasets")

from evalscope import TaskConfig, run_task

port = sys.argv[1]
work_dir = sys.argv[2]

task = TaskConfig(
    model="qwen3.5",
    model_id="qwen3.5",
    api_url=f"http://localhost:{port}/v1/chat/completions",
    api_key="EMPTY",
    datasets=["mmlu_pro"],
    dataset_args={
        "mmlu_pro": {},
    },
    eval_type="openai_api",
    eval_batch_size=32,
    generation_config={
        "batch_size": 32,
        "max_tokens": 8192,
        "n": 1,
        "stream": True,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "do_sample": True,
        "timeout": 600,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    work_dir=work_dir,
    seed=42,
)
run_task(task)
