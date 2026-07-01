#!/usr/bin/env bash
set -euo pipefail

cp -n config.example.env config.env

# 编辑 config.env 后运行：
python3 run_demo.py

# 不重新调用 API，只重新后处理和评估：
# python3 run_demo.py --skip-api
