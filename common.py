from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEMO_ROOT = Path(__file__).resolve().parent

PROMPT_ROOT = DEMO_ROOT / "Prompt"
SYSTEM_PROMPT = PROMPT_ROOT / "SYSTEM_PROMPT.md"
USER_PROMPT_DIR = PROMPT_ROOT / "User_Prompt"
DEFAULT_USER_PROMPT = USER_PROMPT_DIR / "A0_Control_Direct.md"
IMAGE_PATH = DEMO_ROOT / "Tongue-250626-1000-0008.jpg"
GOLD_PATH = DEMO_ROOT / "gold_standard.xlsx"
LABEL_MAPPING_PATH = DEMO_ROOT / "标签映射表.xlsx"

OUTPUT_ROOT = DEMO_ROOT / "output"
MAIN_OUTPUT = OUTPUT_ROOT / "main"
POSTPROCESS_OUTPUT = OUTPUT_ROOT / "postprocess"
EVALUATE_OUTPUT = OUTPUT_ROOT / "evaluate"

def required_files(require_config: bool = False) -> list[Path]:
    files = [SYSTEM_PROMPT, *sorted(USER_PROMPT_DIR.glob("*.md")), IMAGE_PATH, GOLD_PATH, LABEL_MAPPING_PATH]
    if require_config:
        files.append(DEMO_ROOT / "config.env")
    return files


def check_required_files(require_config: bool = False) -> None:
    missing = [path for path in required_files(require_config) if not path.exists()]
    if missing:
        text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"demo 缺少必要文件:\n{text}")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def safe_name(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def latest_jsons(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"), key=lambda p: p.stat().st_mtime)


def run_id_from_path(path: Path) -> str:
    parts = path.stem.split("__")
    return parts[-1] if len(parts) >= 3 else path.stem


def latest_batch_jsons(root: Path) -> list[Path]:
    jsons = latest_jsons(root)
    if not jsons:
        return []
    latest_run_id = run_id_from_path(jsons[-1])
    return [path for path in jsons if run_id_from_path(path) == latest_run_id]
