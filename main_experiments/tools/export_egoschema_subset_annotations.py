from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EgoSchema Subset annotations to local JSON.")
    parser.add_argument("--dataset-path", default="lmms-lab/egoschema")
    parser.add_argument("--dataset-name", default="Subset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="reports/egoschema_subset_annotations.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.split)
    rows = [_jsonable(dict(row)) for row in dataset]
    payload = {
        "dataset": f"{args.dataset_path}/{args.dataset_name}/{args.split}",
        "count": len(rows),
        "records": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"dataset: {payload['dataset']}")
    print(f"records: {payload['count']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
