from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    GOLD_PATH,
    LABEL_MAPPING_PATH,
    MAIN_OUTPUT,
    POSTPROCESS_OUTPUT,
    check_required_files,
    latest_batch_jsons,
    read_json,
    safe_name,
    write_json,
)

MUTUAL_EXCLUSIVE_PAIRS = [
    ("嫩舌", "老舌"),
    ("胖舌", "瘦舌"),
    ("薄苔", "厚苔"),
    ("无苔", "薄苔"),
    ("无苔", "厚苔"),
    ("无苔", "滑润苔"),
    ("无苔", "燥苔"),
    ("无苔", "腐腻苔"),
    ("无苔", "剥落苔"),
    ("无苔", "白苔"),
    ("无苔", "黄苔"),
    ("无苔", "灰黑苔"),
]


def load_gold_labels() -> tuple[list[str], dict[str, int]]:
    df = pd.read_excel(GOLD_PATH)
    labels = [str(c) for c in df.columns[1:]]
    support = {label: int(df[label].fillna(0).astype(int).sum()) for label in labels}
    return labels, support


def load_label_map(valid_labels: list[str]) -> dict[str, str]:
    df = pd.read_excel(LABEL_MAPPING_PATH)
    valid = set(valid_labels)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        alias = str(row.get("原标签", "")).strip()
        target = str(row.get("标准标签", "")).strip()
        enabled = str(row.get("是否启用", "")).strip()
        decision = str(row.get("决策(保留/回退)", "")).strip()
        if alias and target and enabled in {"1", "1.0"} and decision == "保留" and target in valid:
            mapping[alias] = target
    return mapping


def normalize_labels(raw: Any, valid_labels: list[str], label_map: dict[str, str]) -> list[str]:
    if isinstance(raw, dict):
        labels = [str(k).strip() for k, v in raw.items() if str(v).strip() in {"1", "1.0", "True", "true"} or v is True]
    elif isinstance(raw, list):
        labels = [str(x).strip() for x in raw]
    else:
        labels = []

    valid = set(valid_labels)
    clean: list[str] = []
    for label in labels:
        label = label_map.get(label, label)
        if label in valid and label not in clean:
            clean.append(label)
    return clean


def resolve_conflicts(labels: list[str], support: dict[str, int]) -> list[str]:
    label_set = set(labels)
    for a, b in MUTUAL_EXCLUSIVE_PAIRS:
        if a in label_set and b in label_set:
            drop = b if support.get(a, 0) >= support.get(b, 0) else a
            label_set.remove(drop)
    return [x for x in labels if x in label_set]


def majority_vote(attempts: list[list[str]], valid_labels: list[str]) -> list[str]:
    if not attempts:
        return []
    return [
        label
        for label in valid_labels
        if sum(1 for attempt in attempts if label in attempt) > len(attempts) / 2
    ]


def process_file(path: Path) -> Path:
    valid_labels, support = load_gold_labels()
    label_map = load_label_map(valid_labels)
    data = read_json(path)

    for item in data.get("results", []):
        attempts: list[list[str]] = []
        for gen in item.get("generations", []):
            parsed = gen.get("parsed_output") if isinstance(gen, dict) else None
            raw_labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
            clean = normalize_labels(raw_labels, valid_labels, label_map)
            clean = resolve_conflicts(clean, support)
            gen["post_labels"] = clean
            gen["parse_success"] = isinstance(parsed, dict)
            if isinstance(parsed, dict):
                parsed["labels"] = clean
                attempts.append(clean)

        voted = resolve_conflicts(majority_vote(attempts, valid_labels), support)
        item["postprocess"] = {
            "attempt_count": len(attempts),
            "voted_labels": voted,
            "ruleset": "label_mapping + mutual_exclusive + majority_vote",
        }

    meta = data.get("experiment_metadata", {})
    model = safe_name(meta.get("model", "unknown_model"))
    prompt = safe_name(meta.get("prompt_name", "unknown_prompt"))
    run_id = safe_name(meta.get("timestamp", path.stem))
    output = POSTPROCESS_OUTPUT / model / prompt / f"postprocess_{model}__{prompt}__{run_id}.json"
    return write_json(output, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="demo: 清洗模型 JSON 并做标签投票")
    parser.add_argument("--input-json", nargs="*", default=None)
    args = parser.parse_args()

    check_required_files()
    inputs = [Path(x) for x in args.input_json] if args.input_json else latest_batch_jsons(MAIN_OUTPUT)
    if not inputs:
        raise SystemExit(f"未找到 main JSON: {MAIN_OUTPUT}")
    outputs = [process_file(path) for path in inputs]
    print("后处理完成:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
