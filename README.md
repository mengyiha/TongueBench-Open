# TongueBench Demo

本项目用于单张舌象图像的多标签识别 demo：调用LMM，解析模型 JSON 输出，做标签后处理，并与金标准表格计算评估指标。

## 项目结构

- `Prompt/`：系统提示词和用户提示词。
- `Tongue-250626-1000-0008.jpg`：demo 输入图片。
- `gold_standard.xlsx`：金标准标签。
- `标签映射表.xlsx`：模型输出标签到标准标签的映射表。
- `api_runner.py`：调用视觉模型，保存原始响应和解析后的 JSON。
- `postprocess_results.py`：标签映射、非法标签过滤、互斥标签处理、多轮多数投票。
- `evaluate_results.py`：对比金标准，输出 JSON/XLSX 指标。
- `run_demo.py`：按顺序执行 API 调用、后处理和评估。

## 环境准备

建议使用 Python 3.9+。

```bash
pip install pandas scikit-learn openai python-dotenv json-repair openpyxl
cp config.example.env config.env
```

编辑 `config.env`：

```env
MODEL_NAME=your-vlm-model-name
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEYS=sk-your-key
```

`OPENAI_BASE_URL` 需要兼容 OpenAI Chat Completions API，并支持图片输入。

## 运行

完整运行：

```bash
python3 run_demo.py
```

指定每张图生成轮数：

```bash
python3 run_demo.py --n-generations 3
```

部分模型或接口即使传入 `--n-generations 3`，也可能因为服务端限制只返回 1 轮。程序会以 API 响应中的 `choices` 数量作为实际返回轮数，并在控制台输出：

```text
请求生成轮数: 3; 实际返回轮数: 1
提示: 当前模型或接口可能限制单次请求的多轮返回数量。
```

主结果 JSON 的 `experiment_metadata` 中会记录：

- `requested_generations`：本次请求的生成轮数。
- `actual_generations`：模型实际返回的生成轮数。

只重新执行后处理和评估：

```bash
python3 run_demo.py --skip-api
```

`--skip-api` 要求 `output/main/` 下已经存在 `api_runner.py` 生成的主结果 JSON。

## 分步运行

```bash
python3 api_runner.py --config config.env --n-generations 3
python3 postprocess_results.py
python3 evaluate_results.py
```

## 输出

- `output/main/`：模型原始返回、每轮 `raw_content`、解析后的 `parsed_output`。
- `output/postprocess/`：清洗后的 `post_labels` 和 `voted_labels`。
- `output/evaluate/`：`metrics_*.xlsx`、`details_*.xlsx`、`report_*.json`。

## 当前试运行结果

本地已从配置好的 `config.example.env` 复制生成 `config.env`，端到端流程已跑通。

本次验证中请求 3 轮，模型实际返回 1 轮，程序已在控制台提示，并在主结果 JSON 中记录：

```json
"requested_generations": 3,
"actual_generations": 1
```

如果只想复跑后处理和评估，可执行：

```bash
python3 run_demo.py --skip-api
```
