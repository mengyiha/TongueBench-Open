from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import DEMO_ROOT, USER_PROMPT_DIR


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=DEMO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="demo: 舌象模型调用、后处理、评估")
    parser.add_argument("--config", default="config.env")
    parser.add_argument("--n-generations", type=int, default=3)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    if not config.is_absolute():
        config = DEMO_ROOT / config
    if not config.exists() and not args.skip_api:
        raise SystemExit("请先执行: cp config.example.env config.env，并填写配置")

    if not args.skip_api:
        user_prompts = sorted(USER_PROMPT_DIR.glob("*.md"))
        if not user_prompts:
            raise SystemExit(f"未找到用户提示词: {USER_PROMPT_DIR}")
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        for user_prompt in user_prompts:
            cmd = [
                sys.executable,
                "api_runner.py",
                "--config",
                str(config),
                "--user-prompt",
                str(user_prompt),
                "--n-generations",
                str(args.n_generations),
                "--run-id",
                run_id,
            ]
            run(cmd)

    run([sys.executable, "postprocess_results.py"])
    run([sys.executable, "evaluate_results.py"])


if __name__ == "__main__":
    main()
