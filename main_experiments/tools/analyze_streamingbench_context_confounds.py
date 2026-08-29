#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PSM_HISTORY_INSTRUCTION = (
    "When historical frames are present, they appear before the six recent frames.\n\n"
)

PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def extract_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


def timestamp_to_seconds(ts: Any) -> float:
    nums = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(ts))]
    if len(nums) >= 3:
        return nums[-3] * 3600.0 + nums[-2] * 60.0 + nums[-1]
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    return nums[0] if nums else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(path), payload.get("config", {}), payload.get("results", [])
        return str(path), {}, payload

    merged = sorted(
        path.glob("streaming_bench_minicpmv46_results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if merged:
        return load_results(merged[0])

    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No saved StreamingBench results found under {path}")
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        rows.extend(read_jsonl(rank_file))
    dedup: dict[Any, dict[str, Any]] = {}
    for row in rows:
        dedup[row.get("_index", row.get("_key"))] = row
    return "\n  ".join(str(item) for item in rank_files), {}, sorted(
        dedup.values(), key=lambda item: int(item.get("_index", 0))
    )


def format_options(options: Any) -> str:
    if isinstance(options, dict):
        values = [str(options[key]).strip() for key in "ABCDE" if key in options]
    elif isinstance(options, list):
        values = [str(item).strip() for item in options]
    else:
        values = []
    formatted: list[str] = []
    for index, option in enumerate(values):
        letter = chr(65 + index)
        if option.startswith(("A.", "B.", "C.", "D.", "E.")):
            formatted.append(option)
        else:
            formatted.append(f"{letter}. {option}")
    return "\n".join(formatted)


def build_prompt(question: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        question=question.get("question", ""),
        options=format_options(question.get("options", [])),
    )


def load_annotations(path: Path) -> dict[int, tuple[dict[str, Any], dict[str, Any], str]]:
    data = json.load(path.open(encoding="utf-8"))
    flat: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for entry in data:
        questions = sorted(entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        for question in questions:
            flat.append((entry, question, build_prompt(question)))
    return {index: item for index, item in enumerate(flat)}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(int(item))
    return out


def as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(float(item))
    return out


def adaptive_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("adaptive")
    if isinstance(value, dict):
        return value
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def baseline_recent_cdas(row: dict[str, Any]) -> dict[str, Any]:
    adaptive = adaptive_metadata(row)
    if adaptive.get("mode") == "recent_sampler_exact_six":
        return adaptive
    baseline_recent = (adaptive.get("baseline_recent_equivalence") or {}).get("baseline_recent") or {}
    cdas = baseline_recent.get("cdas") or {}
    return cdas if isinstance(cdas, dict) else {}


def suffix(values: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    return list(values[-count:])


def rounded(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def row_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def config_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value is None:
        return None
    return str(value)


def row_context_record(
    index: int,
    base: dict[str, Any],
    ours: dict[str, Any],
    ann: tuple[dict[str, Any], dict[str, Any], str] | None,
    base_config: dict[str, Any],
    ours_config: dict[str, Any],
) -> dict[str, Any]:
    _entry, ann_question, base_prompt = ann if ann is not None else ({}, {}, str(base.get("question", "")))
    adaptive = adaptive_metadata(ours)
    memory_ids = as_int_list(adaptive.get("memory_chunk_ids"))
    base_recent_meta = baseline_recent_cdas(base)
    ours_recent_meta = baseline_recent_cdas(ours)
    base_final_ids = as_int_list(base.get("final_chunk_ids"))
    ours_recent_ids = as_int_list(adaptive.get("recent_chunk_ids"))
    ours_final_ids = as_int_list(ours.get("final_chunk_ids"))
    recent_count = int(base_recent_meta.get("selected_frame_count") or base.get("num_frames") or 0)
    if memory_ids and len(ours_final_ids) >= recent_count > 0:
        ours_final_recent_ids = suffix(ours_final_ids, recent_count)
    else:
        ours_final_recent_ids = list(ours_final_ids)

    base_recent_hashes = as_str_list(base_recent_meta.get("recent_frame_hashes") or adaptive_metadata(base).get("recent_frame_hashes"))
    ours_recent_hashes = as_str_list(ours_recent_meta.get("recent_frame_hashes") or adaptive.get("recent_frame_hashes"))
    base_recent_timestamps = as_float_list(base_recent_meta.get("recent_frame_timestamps") or base_recent_meta.get("selected_timestamps"))
    ours_recent_timestamps = as_float_list(ours_recent_meta.get("recent_frame_timestamps") or ours_recent_meta.get("selected_timestamps"))
    base_recent_indices = as_int_list(base_recent_meta.get("recent_frame_indices") or base_recent_meta.get("selected_frame_indices"))
    ours_recent_indices = as_int_list(ours_recent_meta.get("recent_frame_indices") or ours_recent_meta.get("selected_frame_indices"))

    pixel_identity_available = bool(base_recent_hashes and ours_recent_hashes)
    pixel_identical = pixel_identity_available and base_recent_hashes == ours_recent_hashes
    timestamp_equivalent = bool(base_recent_timestamps and ours_recent_timestamps) and (
        rounded(base_recent_timestamps) == rounded(ours_recent_timestamps)
    )
    frame_index_equivalent = bool(base_recent_indices and ours_recent_indices) and base_recent_indices == ours_recent_indices

    base_prompt_final = base_prompt
    ours_prompt_final = f"{PSM_HISTORY_INSTRUCTION}{base_prompt}" if memory_ids else base_prompt
    prompt_same = base_prompt_final == ours_prompt_final
    visual_sequence_same = (
        pixel_identical
        if pixel_identity_available
        else base_final_ids == ours_final_recent_ids
    )
    frame_count_same = int(base.get("num_frames") or -1) == int(ours.get("num_frames") or -2)
    recent_frame_count_same = int(base_recent_meta.get("selected_frame_count") or len(base_recent_hashes) or len(base_final_ids)) == int(
        ours_recent_meta.get("selected_frame_count") or len(ours_recent_hashes) or len(ours_final_recent_ids)
    )

    compare_keys = ("qa_model", "chunk_duration", "fps", "top_k", "recent_frames_only",
                    "context_time", "frame_selection", "attn_implementation",
                    "downsample_mode", "max_slice_nums")
    generation_param_diffs = {
        key: [config_value(base_config, key), config_value(ours_config, key)]
        for key in compare_keys
        if config_value(base_config, key) != config_value(ours_config, key)
    }
    # The actual final generation method is the same code path in both runs:
    # RecentWindowQAModel.generate_from_frames(..., do_sample=False).
    exact_context_same = (
        visual_sequence_same
        and recent_frame_count_same
        and (not memory_ids)
        and frame_count_same
        and prompt_same
        and not generation_param_diffs.get("downsample_mode")
        and not generation_param_diffs.get("max_slice_nums")
        and not generation_param_diffs.get("attn_implementation")
    )

    base_pred = extract_answer(base.get("response"))
    ours_pred = extract_answer(ours.get("response"))
    same_prediction = base_pred == ours_pred
    if exact_context_same and same_prediction:
        group = "SAME_CONTEXT + SAME_PREDICTION"
    elif exact_context_same:
        group = "SAME_CONTEXT + DIFFERENT_PREDICTION"
    elif same_prediction:
        group = "DIFFERENT_CONTEXT + SAME_PREDICTION"
    else:
        group = "DIFFERENT_CONTEXT + DIFFERENT_PREDICTION"

    memory_added = bool(memory_ids)
    recent_context_changed = (
        not memory_added
        and (
            not visual_sequence_same
            or not recent_frame_count_same
        )
    )
    if pixel_identity_available:
        recent_context_change_reason = (
            "pixel_hash_mismatch"
            if not pixel_identical
            else "recent_frame_count_mismatch"
            if not recent_frame_count_same
            else None
        )
    else:
        recent_context_change_reason = (
            "chunk_id_mismatch_without_hashes"
            if base_final_ids != ours_final_recent_ids
            else "recent_frame_count_mismatch"
            if not recent_frame_count_same
            else None
        )
    if memory_added:
        context_change_type = "memory_added"
    elif recent_context_changed:
        context_change_type = "recent_context_changed"
    elif exact_context_same:
        context_change_type = "exact_context_same"
    else:
        context_change_type = "other_input_difference"

    if not row_correct(base) and row_correct(ours):
        outcome_change = "WC"
    elif row_correct(base) and not row_correct(ours):
        outcome_change = "CW"
    elif row_correct(base) and row_correct(ours):
        outcome_change = "CC"
    else:
        outcome_change = "WW"

    return {
        "question_id": index,
        "key": ours.get("_key") or base.get("_key"),
        "video_id": ours.get("video") or base.get("video"),
        "timestamp": ours.get("time_stamp") or base.get("time_stamp"),
        "category": ours.get("task_type") or base.get("task_type"),
        "question": ours.get("question") or base.get("question"),
        "answer_options": format_options(ann_question.get("options", [])),
        "ground_truth": ours.get("answer_gt") or base.get("answer_gt"),
        "baseline_prediction": base_pred,
        "ours_prediction": ours_pred,
        "baseline_correct": row_correct(base),
        "ours_correct": row_correct(ours),
        "outcome_change": outcome_change,
        "group": group,
        "context_change_type": context_change_type,
        "exact_context_same": exact_context_same,
        "visual_sequence_same": visual_sequence_same,
        "frame_count_same": frame_count_same,
        "recent_frame_count_same": recent_frame_count_same,
        "pixel_identity_available": pixel_identity_available,
        "pixel_identical": pixel_identical,
        "timestamp_equivalent": timestamp_equivalent,
        "frame_index_equivalent": frame_index_equivalent,
        "recent_context_change_reason": recent_context_change_reason,
        "prompt_same": prompt_same,
        "baseline_final_chunk_ids": base_final_ids,
        "ours_recent_chunk_ids": ours_recent_ids,
        "ours_memory_chunk_ids": memory_ids,
        "ours_final_chunk_ids": ours_final_ids,
        "ours_final_recent_chunk_ids": ours_final_recent_ids,
        "baseline_recent_frame_hashes": base_recent_hashes,
        "ours_recent_frame_hashes": ours_recent_hashes,
        "ours_memory_frame_hashes": as_str_list(adaptive.get("memory_frame_hashes")),
        "ours_final_frame_hashes": as_str_list(adaptive.get("final_frame_hashes")),
        "baseline_recent_frame_timestamps": base_recent_timestamps,
        "ours_recent_frame_timestamps": ours_recent_timestamps,
        "baseline_recent_frame_indices": base_recent_indices,
        "ours_recent_frame_indices": ours_recent_indices,
        "baseline_num_frames": base.get("num_frames"),
        "ours_num_frames": ours.get("num_frames"),
        "baseline_prompt_hash": stable_hash(base_prompt_final),
        "ours_prompt_hash": stable_hash(ours_prompt_final),
        "ours_prompt_has_psm_history_instruction": bool(memory_ids),
        "baseline_decode_backend": base.get("decode_backend"),
        "ours_decode_backend": ours.get("decode_backend"),
        "ours_baseline_recent_metadata": (adaptive.get("baseline_recent_equivalence") or {}).get("baseline_recent"),
        "ours_stop_reason": adaptive.get("stop_reason"),
        "ours_final_sufficiency": adaptive.get("final_sufficiency"),
        "ours_num_sufficiency_iterations": adaptive.get("num_sufficiency_iterations"),
        "generation_param_diffs": generation_param_diffs,
        "baseline_raw": base.get("response"),
        "ours_raw": ours.get("response"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()) if flat_rows else [])
        writer.writeheader()
        writer.writerows(flat_rows)


def print_counts(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}")
    for key, value in counter.most_common():
        print(f"{key}: {value}")


def print_same_context_different_prediction(rows: list[dict[str, Any]]) -> None:
    cases = [row for row in rows if row["group"] == "SAME_CONTEXT + DIFFERENT_PREDICTION"]
    print(f"\nSAME_CONTEXT + DIFFERENT_PREDICTION CASES: {len(cases)}")
    for row in cases:
        print("-" * 80)
        print(f"Q{row['question_id']} {row['category']} {row['video_id']} {row['timestamp']}")
        print(row["question"])
        print(f"GT={row['ground_truth']} base={row['baseline_prediction']} ours={row['ours_prediction']}")
        print(f"chunks={row['baseline_final_chunk_ids']} frames={row['baseline_num_frames']}")
        print(f"prompt_same={row['prompt_same']} decode={row['baseline_decode_backend']}->{row['ours_decode_backend']}")
        print(f"psm_iters={row['ours_num_sufficiency_iterations']} stop={row['ours_stop_reason']}")
        print(f"generation_param_diffs={row['generation_param_diffs']}")


def print_recent_context_changed_examples(rows: list[dict[str, Any]], limit: int = 20) -> None:
    cases = [row for row in rows if row["context_change_type"] == "recent_context_changed"]
    print(f"\nFIRST {min(limit, len(cases))} RECENT_CONTEXT_CHANGED CASES")
    for row in cases[:limit]:
        print("-" * 80)
        print(f"Q{row['question_id']} {row['category']} {row['video_id']} {row['timestamp']}")
        print(f"cause={row['recent_context_change_reason']}")
        print("baseline:")
        print(f"  timestamps={row['baseline_recent_frame_timestamps']}")
        print(f"  chunk_ids={row['baseline_final_chunk_ids']}")
        print(f"  frame_indices={row['baseline_recent_frame_indices']}")
        print(f"  num_frames={row['baseline_num_frames']}")
        print(f"  hashes={row['baseline_recent_frame_hashes']}")
        print("PRISM:")
        print(f"  timestamps={row['ours_recent_frame_timestamps']}")
        print(f"  recent_chunk_ids={row['ours_recent_chunk_ids']}")
        print(f"  final_recent_chunk_ids={row['ours_final_recent_chunk_ids']}")
        print(f"  frame_indices={row['ours_recent_frame_indices']}")
        print(f"  num_frames={row['ours_num_frames']}")
        print(f"  hashes={row['ours_recent_frame_hashes']}")
        print(
            "equivalence: "
            f"pixel={row['pixel_identical']} timestamp={row['timestamp_equivalent']} "
            f"frame_index={row['frame_index_equivalent']} prompt={row['prompt_same']}"
        )
        print(f"generation_param_diffs={row['generation_param_diffs']}")


def print_code_path_explanation() -> None:
    print("\nCODE PATH NOTES")
    print("- Baseline and PSM final answers both call RecentWindowQAModel.generate_from_frames(...).")
    print("- generate_from_frames calls model.generate(..., do_sample=False), so decoding is intended to be deterministic.")
    print("- If memory_chunk_ids is non-empty, PSM prepends a history instruction to the prompt.")
    print("- If memory_chunk_ids is empty, PSM final prompt is the original prompt.")
    print("- PSM always performs extra option/sufficiency forward passes before final generation.")
    print("- Same chunk IDs do not prove pixel-identical frames; new exact-recent runs save ordered RGB frame hashes.")
    print("- When hashes are present, SAME_CONTEXT uses ordered pixel identity plus frame count/prompt/config checks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare StreamingBench baseline vs PSM final input confounds.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    base_source, base_config, base_rows = load_results(args.baseline)
    ours_source, ours_config, ours_rows = load_results(args.ours)
    annotations = load_annotations(args.annotations)
    base_by_index = {int(row["_index"]): row for row in base_rows if row.get("_index") is not None}
    ours_by_index = {int(row["_index"]): row for row in ours_rows if row.get("_index") is not None}
    common = sorted(set(base_by_index) & set(ours_by_index))
    rows = [
        row_context_record(
            index,
            base_by_index[index],
            ours_by_index[index],
            annotations.get(index),
            base_config,
            ours_config,
        )
        for index in common
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "streamingbench_context_confounds.json"
    csv_path = args.out_dir / "streamingbench_context_confounds.csv"
    summary_path = args.out_dir / "streamingbench_context_confounds_summary.json"

    summary = {
        "baseline_source": base_source,
        "ours_source": ours_source,
        "matched_samples": len(rows),
        "group_counts": Counter(row["group"] for row in rows),
        "context_change_type_counts": Counter(row["context_change_type"] for row in rows),
        "equivalence_counts": {
            "pixel_identity_available": sum(1 for row in rows if row["pixel_identity_available"]),
            "pixel_identical": sum(1 for row in rows if row["pixel_identical"]),
            "timestamp_equivalent": sum(1 for row in rows if row["timestamp_equivalent"]),
            "frame_index_equivalent": sum(1 for row in rows if row["frame_index_equivalent"]),
            "recent_context_change_reasons": Counter(
                row["recent_context_change_reason"]
                for row in rows
                if row["recent_context_change_reason"]
            ),
        },
        "outcome_by_context_change_type": {
            change_type: Counter(row["outcome_change"] for row in rows if row["context_change_type"] == change_type)
            for change_type in sorted({row["context_change_type"] for row in rows})
        },
    }
    json_path.write_text(
        json.dumps(
            {
                **summary,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"Baseline source: {base_source}")
    print(f"Our source: {ours_source}")
    print(f"Matched samples: {len(rows)}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print_counts("FOUR GROUPS", Counter(row["group"] for row in rows))
    print_counts("CONTEXT CHANGE TYPES", Counter(row["context_change_type"] for row in rows))
    print_counts(
        "RECENT CONTEXT CHANGE REASONS",
        Counter(row["recent_context_change_reason"] for row in rows if row["recent_context_change_reason"]),
    )
    print("\nEQUIVALENCE COUNTS")
    print(f"pixel_identity_available: {sum(1 for row in rows if row['pixel_identity_available'])}")
    print(f"pixel_identical: {sum(1 for row in rows if row['pixel_identical'])}")
    print(f"timestamp_equivalent: {sum(1 for row in rows if row['timestamp_equivalent'])}")
    print(f"frame_index_equivalent: {sum(1 for row in rows if row['frame_index_equivalent'])}")

    print("\nOUTCOME BY CONTEXT CHANGE TYPE")
    print("context_change_type,CC,WC,CW,WW,prediction_unchanged,net_accuracy_effect")
    for change_type in sorted({row["context_change_type"] for row in rows}):
        subset = [row for row in rows if row["context_change_type"] == change_type]
        outcomes = Counter(row["outcome_change"] for row in subset)
        unchanged_pred = sum(1 for row in subset if row["baseline_prediction"] == row["ours_prediction"])
        print(
            f"{change_type},{outcomes['CC']},{outcomes['WC']},{outcomes['CW']},{outcomes['WW']},"
            f"{unchanged_pred},{outcomes['WC'] - outcomes['CW']}"
        )

    print_same_context_different_prediction(rows)
    print_recent_context_changed_examples(rows)
    print_code_path_explanation()


if __name__ == "__main__":
    main()
