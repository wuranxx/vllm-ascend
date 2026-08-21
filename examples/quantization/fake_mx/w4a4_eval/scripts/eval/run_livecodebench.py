#!/usr/bin/env python3
"""LiveCodeBench 评测脚本（0-shot，需 docker sandbox）。

用法: python run_livecodebench.py <port> <work_dir>
示例: python run_livecodebench.py 8001 ./outputs/bf16_livecodebench

环境要求:
  conda activate evalscope  (evalscope 1.10.0, pyarrow 19.0.1, ms-enclave 0.0.8)
  需有 docker 和 python:3.11-slim 镜像
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
    datasets=["live_code_bench"],
    dataset_args={
        "live_code_bench": {
            "subset_list": ["release_latest"],
            "extra_params": {},
        }
    },
    eval_type="openai_api",
    eval_batch_size=8,
    generation_config={
        "batch_size": 8,
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
    use_sandbox=True,
    sandbox={
        "enabled": True,
        "engine": "docker",
        "default_config": {
            "image": "python:3.11-slim",
        },
    },
    work_dir=work_dir,
    seed=42,
)
run_task(task)
