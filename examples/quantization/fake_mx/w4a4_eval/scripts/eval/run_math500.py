#!/usr/bin/env python3
"""通用 MATH-500 评测脚本。

用法: python run_math500.py <port> <work_dir>
示例: python run_math500.py 8001 ./outputs/rtn_attn-only_w4a4
"""

import sys

from evalscope import TaskConfig, run_task

port = sys.argv[1]
work_dir = sys.argv[2]

task = TaskConfig(
    model="qwen3.5",
    model_id="qwen3.5",
    api_url=f"http://localhost:{port}/v1/chat/completions",
    api_key="EMPTY",
    datasets=["math_500"],
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
