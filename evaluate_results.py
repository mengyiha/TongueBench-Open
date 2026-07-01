from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

from common import (
    EVALUATE_OUTPUT,
    GOLD_PATH,
    POSTPROCESS_OUTPUT,
    check_required_files,
    latest_batch_jsons,
    read_json,
    safe_name,
)


def load_gold() -> tuple[list[str], dict[str, list[str]]]:
    df = pd.read_excel(GOLD_PATH)
    image_col = df.columns[0]
    labels = [str(c) for c in df.columns[1:]]
    gold: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        image_id = Path(str(row[image_col])).stem
        gold[image_id] = [
            label
            for label in labels
            if (not pd.isna(row[label])) and int(row[label]) == 1
        ]
        if image_id.endswith("-1000"):
            gold[image_id[:-5]] = gold[image_id]
        else:
            gold[f"{image_id}-1000"] = gold[image_id]
    return labels, gold


def to_binary(labels: list[str], all_labels: list[str]) -> list[int]:
    label_set = set(labels)
    return [1 if label in label_set else 0 for label in all_labels]


def collect_predictions(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in data.get("results", []):
        image_id = str(item.get("image", "")).strip()
        if not image_id:
            continue
        voted = item.get("postprocess", {}).get("voted_labels", [])
        rows.append({"image": image_id, "strategy": "voted", "labels": voted if isinstance(voted, list) else []})
        for gen in item.get("generations", []):
            labels = gen.get("post_labels", []) if isinstance(gen, dict) else []
            rows.append(
                {
                    "image": image_id,
                    "strategy": f"attempt_{int(gen.get('index', 0)) + 1}",
                    "labels": labels if isinstance(labels, list) else [],
                }
            )
    return rows


def evaluate_file(path: Path) -> dict[str, Path]:
    labels, gold = load_gold()
    data = read_json(path)
    meta = data.get("experiment_metadata", {})

    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[str, list[list[int]]]] = {}
    for pred in collect_predictions(data):
        image_id = pred["image"]
        if image_id not in gold:
            continue
        y_true = to_binary(gold[image_id], labels)
        y_pred = to_binary(pred["labels"], labels)
        strategy = pred["strategy"]
        vectors.setdefault(strategy, {"true": [], "pred": []})
        vectors[strategy]["true"].append(y_true)
        vectors[strategy]["pred"].append(y_pred)
        rows.append(
            {
                "Image_ID": image_id,
                "Strategy": strategy,
                "Exact_Match": int(y_true == y_pred),
                "Hamming_Loss": hamming_loss([y_true], [y_pred]),
                **dict(zip(labels, y_pred)),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for strategy, values in sorted(vectors.items()):
        y_true = values["true"]
        y_pred = values["pred"]
        summary_rows.append(
            {
                "Model": meta.get("model", ""),
                "Prompt": meta.get("prompt_name", ""),
                "Timestamp": meta.get("timestamp", ""),
                "Strategy": strategy,
                "Samples": len(y_true),
                "Exact_Match_Ratio": accuracy_score(y_true, y_pred),
                "Micro_F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "Samples_F1": f1_score(y_true, y_pred, average="samples", zero_division=0),
                "Hamming_Loss": hamming_loss(y_true, y_pred),
                "Jaccard_Samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
            }
        )

    model = safe_name(meta.get("model", "unknown_model"))
    prompt = safe_name(meta.get("prompt_name", "unknown_prompt"))
    run_id = safe_name(meta.get("timestamp", path.stem))
    out_dir = EVALUATE_OUTPUT / model / prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"metrics_{model}__{prompt}__{run_id}.xlsx"
    detail_path = out_dir / f"details_{model}__{prompt}__{run_id}.xlsx"
    report_path = out_dir / f"report_{model}__{prompt}__{run_id}.json"

    pd.DataFrame(summary_rows).to_excel(summary_path, index=False)
    pd.DataFrame(rows).to_excel(detail_path, index=False)
    report_path.write_text(pd.DataFrame(summary_rows).to_json(force_ascii=False, orient="records", indent=2), encoding="utf-8")
    return {"summary": summary_path, "detail": detail_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="demo: 与金标准对比评估")
    parser.add_argument("--input-json", nargs="*", default=None)
    args = parser.parse_args()

    check_required_files()
    inputs = [Path(x) for x in args.input_json] if args.input_json else latest_batch_jsons(POSTPROCESS_OUTPUT)
    if not inputs:
        raise SystemExit(f"未找到 postprocess JSON: {POSTPROCESS_OUTPUT}")

    for path in inputs:
        outputs = evaluate_file(path)
        print("评估完成:", path)
        for value in outputs.values():
            print(value)


if __name__ == "__main__":
    main()
