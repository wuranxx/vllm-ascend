"""Run a Qwen3.5 fake-MX prefill/decode smoke test without modifying the checkpoint."""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_model(model: Path, quant_config: Path, staging_root: Path) -> Path:
    staged_model = staging_root / "model"
    staged_model.mkdir()
    for source in model.iterdir():
        if source.name == "quant_model_description.json":
            continue
        os.symlink(source.resolve(), staged_model / source.name, target_is_directory=source.is_dir())
    (staged_model / "quant_model_description.json").write_bytes(quant_config.read_bytes())
    return staged_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--quant-config", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--prompt", default="Hello world, this is a fake MX validation prompt.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    model = args.model.resolve(strict=True)
    quant_config = args.quant_config.resolve(strict=True)
    json.loads(quant_config.read_text(encoding="utf-8"))

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vllm-fake-mx-") as tmp:
        staged_model = _stage_model(model, quant_config, Path(tmp))
        llm = LLM(
            model=str(staged_model),
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=args.seed,
        )
        engine_ready_seconds = time.perf_counter() - started
        generation_started = time.perf_counter()
        outputs = llm.generate(
            [args.prompt],
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed),
        )
        generation_seconds = time.perf_counter() - generation_started

    candidate = outputs[0].outputs[0]
    result = {
        "source_model": str(model),
        "quant_config": str(quant_config),
        "quant_config_sha256": _file_sha256(quant_config),
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt": args.prompt,
        "engine_ready_seconds": engine_ready_seconds,
        "generation_seconds": generation_seconds,
        "generated_text": candidate.text,
        "generated_token_ids": list(candidate.token_ids),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
