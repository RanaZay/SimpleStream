#!/usr/bin/env python3
"""Download/extract StreamingBench Real-Time videos one archive at a time."""

from __future__ import annotations

import csv
import ast
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "mjuicem/StreamingBench"
ARCHIVES = [
    "Real-Time Visual Understanding_1-50.zip",
    "Real-Time Visual Understanding_51-100.zip",
    "Real-Time Visual Understanding_101-150.zip",
    "Real-Time Visual Understanding_151-200.zip",
    "Real-Time Visual Understanding_201-250.zip",
    "Real-Time Visual Understanding_251-300.zip",
    "Real-Time Visual Understanding_301-350.zip",
    "Real-Time Visual Understanding_351-400.zip",
    "Real-Time Visual Understanding_401-450.zip",
    "Real-Time Visual Understanding_451-500.zip",
]


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def build_annotations(root: Path) -> None:
    anno_csv = root / "data/streamingbench/annotations/Real_Time_Visual_Understanding.csv"
    anno_json = root / "data/streamingbench/questions_real.json"
    by_sample: dict[str, dict[str, object]] = {}

    def normalise_options(value: str) -> list[str]:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [item.strip() for item in text.split(";") if item.strip()]
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]

    with anno_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row.get("question_id", "")
            if "_sample_" not in qid:
                continue
            sample_id = qid.split("_sample_", 1)[1].rsplit("_", 1)[0]
            video_name = f"sample_{sample_id}_real.mp4"
            question = {
                "question_id": qid,
                "task_type": row.get("task_type", ""),
                "question": row.get("question", ""),
                "time_stamp": row.get("time_stamp", ""),
                "answer": row.get("answer", ""),
                "options": normalise_options(row.get("options", "")),
                "frames_required": row.get("frames_required", ""),
                "temporal_clue_type": row.get("temporal_clue_type", ""),
            }
            entry = by_sample.setdefault(
                video_name,
                {
                    "video_path": video_name,
                    "video_categories": "Real-Time Visual Understanding",
                    "questions": [],
                },
            )
            entry["questions"].append(question)  # type: ignore[index]

    data = []
    for video_name in sorted(by_sample, key=lambda name: int(name.split("_")[1])):
        entry = by_sample[video_name]
        entry["questions"].sort(key=lambda item: item.get("time_stamp", ""))  # type: ignore[index,union-attr]
        data.append(entry)

    anno_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    question_count = sum(len(entry["questions"]) for entry in data)  # type: ignore[arg-type]
    print(f"[annotations] wrote {anno_json} videos={len(data)} questions={question_count}", flush=True)


def main() -> None:
    root = Path.cwd()
    zip_dir = root / "data/hf_streamingbench_zips"
    video_dir = root / "data/streamingbench/videos"
    zip_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[start] free_gb={free_gb(root):.1f} "
        f"existing_videos={len(list(video_dir.glob('sample_*_real.mp4')))}",
        flush=True,
    )

    for archive in ARCHIVES:
        print(f"\n[archive] {archive} free_gb_before={free_gb(root):.1f}", flush=True)
        local = zip_dir / archive
        if not local.exists():
            downloaded = hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=archive,
                local_dir=zip_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            local = Path(downloaded)

        print(f"[archive] local={local} size_gb={local.stat().st_size / 1e9:.2f}", flush=True)
        extracted = 0
        skipped = 0
        with zipfile.ZipFile(local) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.startswith("__MACOSX/") or not name.endswith("/video.mp4"):
                    continue
                sample_dir = next((part for part in Path(name).parts if part.startswith("sample_")), None)
                if sample_dir is None:
                    continue
                try:
                    sample_id = int(sample_dir.split("_", 1)[1])
                except ValueError:
                    continue
                out_path = video_dir / f"sample_{sample_id}_real.mp4"
                if out_path.exists() and out_path.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                tmp_path = out_path.with_suffix(".mp4.tmp")
                with zf.open(info) as src, tmp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                tmp_path.replace(out_path)
                extracted += 1

        print(
            f"[archive] extracted={extracted} skipped={skipped} "
            f"videos_now={len(list(video_dir.glob('sample_*_real.mp4')))} "
            f"free_gb_after_extract={free_gb(root):.1f}",
            flush=True,
        )
        local.unlink()
        print(f"[archive] deleted_zip={local} free_gb_after_delete={free_gb(root):.1f}", flush=True)

    build_annotations(root)
    missing = [idx for idx in range(1, 501) if not (video_dir / f"sample_{idx}_real.mp4").exists()]
    print(
        f"[done] videos={len(list(video_dir.glob('sample_*_real.mp4')))} "
        f"missing_count={len(missing)} missing_first={missing[:20]} free_gb={free_gb(root):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
