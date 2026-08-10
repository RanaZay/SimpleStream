from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _video_cache_dir() -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    return os.path.join(hf_home, "egoschema", "videos")


def _video_id(raw: Any) -> str:
    text = str(raw).strip()
    if text.lower().endswith(".mp4"):
        text = Path(text).stem
    return text


def _exists(video_dir: Path, video_id: str) -> bool:
    return (video_dir / f"{video_id}.mp4").exists() or (video_dir / f"{video_id}.MP4").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="List and verify EgoSchema Subset video IDs.")
    parser.add_argument("--dataset-path", default="lmms-lab/egoschema")
    parser.add_argument("--dataset-name", default="Subset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-dir", default=_video_cache_dir())
    parser.add_argument("--out", default="reports/egoschema_subset_video_ids.json")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.split)
    video_ids = []
    for row in dataset:
        item = dict(row)
        video_ids.append(_video_id(item.get("video_idx", item.get("video_id", item.get("video")))))
    video_ids = sorted(set(video_ids))
    video_dir = Path(args.video_dir)
    missing = [video_id for video_id in video_ids if not _exists(video_dir, video_id)]
    payload = {
        "dataset": f"{args.dataset_path}/{args.dataset_name}/{args.split}",
        "count": len(video_ids),
        "video_dir": str(video_dir),
        "present": len(video_ids) - len(missing),
        "missing": len(missing),
        "video_ids": video_ids,
        "missing_video_ids": missing,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"dataset: {payload['dataset']}")
    print(f"video IDs: {payload['count']}")
    print(f"video dir: {payload['video_dir']}")
    print(f"present: {payload['present']}")
    print(f"missing: {payload['missing']}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
