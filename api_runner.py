from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from json_repair import repair_json
from openai import OpenAI

from common import (
    DEFAULT_USER_PROMPT,
    IMAGE_PATH,
    MAIN_OUTPUT,
    SYSTEM_PROMPT,
    check_required_files,
    load_text,
    safe_name,
    write_json,
)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(repair_json(text))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def call_model(
    client: OpenAI,
    model: str,
    image_b64: str,
    system_prompt: str,
    user_prompt: str,
    n_generations: int,
    timeout: float,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ],
        n=n_generations,
        stream=False,
        timeout=timeout,
    )
    return response.model_dump()


def run_single(
    client: OpenAI,
    model: str,
    user_prompt_path: Path,
    run_id: str,
    n_generations: int,
    timeout: float,
) -> tuple[Path, int]:
    prompt_name = user_prompt_path.stem
    raw_response = call_model(
        client=client,
        model=model,
        image_b64=encode_image(IMAGE_PATH),
        system_prompt=load_text(SYSTEM_PROMPT),
        user_prompt=load_text(user_prompt_path),
        n_generations=n_generations,
        timeout=timeout,
    )
    choices = raw_response.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    actual_generations = len(choices)

    generations: list[dict[str, Any]] = []
    for idx, choice in enumerate(choices):
        content = str(choice.get("message", {}).get("content") or "")
        parsed = parse_json(content)
        generations.append(
            {
                "index": idx,
                "raw_content": content,
                "parsed_output": parsed,
                "parse_success": parsed is not None,
            }
        )

    result = {
        "experiment_metadata": {
            "timestamp": run_id,
            "model": model,
            "system_prompt": str(SYSTEM_PROMPT.relative_to(Path(__file__).resolve().parent)),
            "user_prompt": str(user_prompt_path.relative_to(Path(__file__).resolve().parent)),
            "prompt_name": prompt_name,
            "n_generations": n_generations,
            "requested_generations": n_generations,
            "actual_generations": actual_generations,
            "image_count": 1,
        },
        "results": [
            {
                "image": IMAGE_PATH.stem,
                "status": "success",
                "raw_response": raw_response,
                "generations": generations,
            }
        ],
    }
    output = (
        MAIN_OUTPUT
        / safe_name(model)
        / f"main_{safe_name(model)}__{safe_name(prompt_name)}__{run_id}.json"
    )
    return write_json(output, result), actual_generations


def main() -> None:
    parser = argparse.ArgumentParser(description="demo: 单张舌象图片模型调用")
    parser.add_argument("--config", default="config.env")
    parser.add_argument("--user-prompt", default=str(DEFAULT_USER_PROMPT))
    parser.add_argument("--n-generations", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    user_prompt_path = Path(args.user_prompt)
    if not user_prompt_path.is_absolute():
        user_prompt_path = Path(__file__).resolve().parent / user_prompt_path
    check_required_files()
    if not config_path.exists():
        raise SystemExit(f"配置文件不存在: {config_path}")
    if not user_prompt_path.exists():
        raise SystemExit(f"用户提示词不存在: {user_prompt_path}")

    config = dotenv_values(config_path)
    model = str(config.get("MODEL_NAME") or "").strip()
    base_url = str(config.get("OPENAI_BASE_URL") or "").strip()
    api_key = str(config.get("OPENAI_API_KEYS") or config.get("OPENAI_API_KEY") or "").split(",")[0].strip()
    if not model or not base_url or not api_key:
        raise SystemExit("请在 config.env 中填写 MODEL_NAME、OPENAI_BASE_URL、OPENAI_API_KEYS")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    client = OpenAI(base_url=base_url, api_key=api_key)
    output, actual_generations = run_single(client, model, user_prompt_path, run_id, args.n_generations, args.timeout)
    print(f"用户提示词: {user_prompt_path.name}")
    print(f"请求生成轮数: {args.n_generations}; 实际返回轮数: {actual_generations}")
    if actual_generations != args.n_generations:
        print("提示: 当前模型或接口可能限制单次请求的多轮返回数量。")
    print(f"API 调用完成: {output}")


if __name__ == "__main__":
    main()
