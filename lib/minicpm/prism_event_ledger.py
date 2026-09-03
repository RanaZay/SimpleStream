from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from lib.minicpm.prism_retrieval_variants import chunk_bounds, chunk_id, representative_frame


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


_MOVIE_CLIP_PATTERNS = (
    "movie clip",
    "movie clips",
    "clips been inserted",
    "clip inserted",
)


def is_cumulative_count_question(question: str, options: list[dict[str, str]] | None = None) -> bool:
    text = re.sub(r"\s+", " ", str(question).lower()).strip()
    has_count = text.startswith("how many") or "how many " in text or "number of" in text
    has_history = any(marker in text for marker in ("so far", "in total", "total", "have been", "has been"))
    if not has_count or not has_history:
        return False
    if options is None:
        return True
    values = numeric_options(options)
    return bool(values) and len(values) >= max(2, min(4, len(options)))


def count_target_prompts(question: str) -> dict[str, list[str]]:
    text = re.sub(r"\s+", " ", str(question).lower()).strip()
    if any(pattern in text for pattern in _MOVIE_CLIP_PATTERNS):
        return {
            "positive": [
                "an inserted movie clip is visible",
                "a movie clip or cutaway video is shown",
                "the video shows inserted movie footage",
                "a clip from a movie appears on screen",
            ],
            "negative": [
                "a person is explaining or talking to camera",
                "no inserted movie clip is visible",
                "only the speaker or presenter is visible",
                "a static talking-head explanation",
            ],
        }

    target = semantic_count_query(question, numeric_option_mode=True)
    return {
        "positive": [
            target,
            f"visible instance of {target}",
            f"the counted object or event: {target}",
        ],
        "negative": [
            "unrelated background",
            "no relevant counted object or event",
        ],
    }


def score_chunks_for_count_target(
    scorer: EventLedgerClipScorer,
    chunks: list[Any],
    question: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = [representative_frame(chunk) for chunk in chunks]
    if not frames:
        return [], {"prompts": count_target_prompts(question)}
    image_embeddings = scorer.image_embeddings(frames)
    prompts = count_target_prompts(question)
    positive = prompts["positive"]
    negative = prompts["negative"]
    text_embeddings = scorer.text_embeddings(positive + negative)
    scores = image_embeddings @ text_embeddings.T
    pos_scores = torch.max(scores[:, : len(positive)], dim=1).values
    neg_scores = torch.max(scores[:, len(positive) :], dim=1).values if negative else torch.zeros_like(pos_scores)
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        start, end = chunk_bounds(chunk)
        pos = float(pos_scores[index].detach().cpu())
        neg = float(neg_scores[index].detach().cpu())
        rows.append(
            {
                "chunk_id": int(chunk_id(chunk)),
                "start_time": float(start),
                "end_time": float(end),
                "timestamp": float(chunk_timestamp(chunk)),
                "positive_score": pos,
                "negative_score": neg,
                "ledger_score": pos - neg,
                "raw_relevance": pos,
            }
        )
    return rows, {"prompts": prompts}


def cluster_count_evidence(
    scored_chunks: list[dict[str, Any]],
    *,
    score_threshold: float,
    merge_gap_seconds: float,
    min_positive_run: int = 1,
) -> list[dict[str, Any]]:
    positives = [row for row in scored_chunks if float(row["ledger_score"]) >= float(score_threshold)]
    positives = sorted(positives, key=lambda item: (float(item["start_time"]), float(item["end_time"])))
    clusters: list[dict[str, Any]] = []
    for row in positives:
        if not clusters or float(row["start_time"]) - float(clusters[-1]["end_time"]) > float(merge_gap_seconds):
            clusters.append(
                {
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "member_chunk_ids": [int(row["chunk_id"])],
                    "member_scores": [float(row["ledger_score"])],
                    "member_relevances": [float(row["raw_relevance"])],
                    "peak_score": float(row["ledger_score"]),
                    "peak_relevance": float(row["raw_relevance"]),
                }
            )
        else:
            cluster = clusters[-1]
            cluster["end_time"] = max(float(cluster["end_time"]), float(row["end_time"]))
            cluster["member_chunk_ids"].append(int(row["chunk_id"]))
            cluster["member_scores"].append(float(row["ledger_score"]))
            cluster["member_relevances"].append(float(row["raw_relevance"]))
            cluster["peak_score"] = max(float(cluster["peak_score"]), float(row["ledger_score"]))
            cluster["peak_relevance"] = max(float(cluster["peak_relevance"]), float(row["raw_relevance"]))
    if int(min_positive_run) > 1:
        clusters = [cluster for cluster in clusters if len(cluster["member_chunk_ids"]) >= int(min_positive_run)]
    for index, cluster in enumerate(clusters):
        scores = cluster["member_scores"]
        cluster["event_id"] = index
        cluster["mean_score"] = float(sum(scores) / len(scores)) if scores else 0.0
    return clusters


def targeted_count_ledger(
    scorer: EventLedgerClipScorer,
    chunks: list[Any],
    question: str,
    options: list[dict[str, str]],
    *,
    score_threshold: float,
    merge_gap_seconds: float,
    min_positive_run: int = 1,
) -> dict[str, Any]:
    scored_chunks, meta = score_chunks_for_count_target(scorer, chunks, question)
    clusters = cluster_count_evidence(
        scored_chunks,
        score_threshold=score_threshold,
        merge_gap_seconds=merge_gap_seconds,
        min_positive_run=min_positive_run,
    )
    count = len(clusters)
    mapped = option_for_count(count, options)
    return {
        "ledger_count": int(count),
        "mapped_option": mapped,
        "clusters": clusters,
        "scored_chunks": scored_chunks,
        "score_threshold": float(score_threshold),
        "merge_gap_seconds": float(merge_gap_seconds),
        "min_positive_run": int(min_positive_run),
        **meta,
    }


@dataclass
class LedgerEvent:
    event_id: int
    start_time: float
    end_time: float
    member_chunk_ids: list[int]
    member_timestamps: list[float]
    mean_clip_embedding: torch.Tensor
    peak_change_chunk_id: int
    peak_change_score: float
    representative_chunk_id: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _clip_device() -> str:
    value = os.environ.get("MINICPM_EVENT_LEDGER_CLIP_DEVICE", "").strip()
    if value:
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _clip_model_name() -> str:
    return os.environ.get("MINICPM_EVENT_LEDGER_CLIP_MODEL", "openai/clip-vit-base-patch32")


def _as_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Could not extract CLIP feature tensor from {type(output).__name__}")


class EventLedgerClipScorer:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name or _clip_model_name()
        self.device = device or _clip_device()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.text_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def image_embeddings(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 0), device=self.device)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = _as_feature_tensor(self.model.get_image_features(**inputs)).float()
        return torch.nn.functional.normalize(output, dim=-1)

    @torch.inference_mode()
    def text_embeddings(self, texts: list[str]) -> torch.Tensor:
        missing = [text for text in texts if text not in self.text_cache]
        if missing:
            inputs = self.processor(text=missing, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = _as_feature_tensor(self.model.get_text_features(**inputs)).float()
            output = torch.nn.functional.normalize(output, dim=-1)
            for text, vector in zip(missing, output.detach().cpu()):
                self.text_cache[text] = vector
        return torch.stack([self.text_cache[text] for text in texts], dim=0).to(self.device)


def chunk_timestamp(chunk: Any) -> float:
    start, end = chunk_bounds(chunk)
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return 0.5 * (min(numeric) + max(numeric))
    return 0.5 * (float(start) + float(end))


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(float(value) - low) / (high - low) for value in values]


def _visual_change_scores(chunks: list[Any]) -> list[float]:
    try:
        from lib.minicpm import adaptive as adaptive_mod

        config = SimpleNamespace(dedup_resize=int(os.environ.get("MINICPM_ADAPTIVE_DEDUP_RESIZE", "64")))
        bank = adaptive_mod._build_online_memory_bank(chunks, config)
        return [float(entry.get("event_change_norm", 0.0)) for entry in bank]
    except Exception:
        return [0.0 for _ in chunks]


def event_boundary_scores(
    chunks: list[Any],
    image_embeddings: torch.Tensor,
    visual_change_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    clip_weight = _env_float("MINICPM_EVENT_LEDGER_CLIP_CHANGE_WEIGHT", 0.70)
    visual_weight = _env_float("MINICPM_EVENT_LEDGER_VISUAL_CHANGE_WEIGHT", 0.30)
    visual = visual_change_scores if visual_change_scores is not None else _visual_change_scores(chunks)
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            clip_change = 0.0
        else:
            cosine = float(torch.dot(image_embeddings[index], image_embeddings[index - 1]).detach().cpu())
            clip_change = max(0.0, min(2.0, 1.0 - cosine))
        visual_change = float(visual[index]) if index < len(visual) else 0.0
        boundary = clip_weight * clip_change + visual_weight * visual_change
        rows.append(
            {
                "chunk_id": int(chunk_id(chunk)),
                "timestamp": float(chunk_timestamp(chunk)),
                "clip_change": float(clip_change),
                "visual_change": float(visual_change),
                "boundary_score": float(boundary),
            }
        )
    return rows


def segment_events(chunks: list[Any], image_embeddings: torch.Tensor, boundary_rows: list[dict[str, Any]]) -> list[LedgerEvent]:
    if not chunks:
        return []
    threshold = _env_float("MINICPM_EVENT_LEDGER_BOUNDARY_THRESHOLD", 0.35)
    max_event_seconds = _env_float("MINICPM_EVENT_LEDGER_MAX_EVENT_SECONDS", 5.0)
    min_event_seconds = _env_float("MINICPM_EVENT_LEDGER_MIN_EVENT_SECONDS", 1.0)

    events: list[LedgerEvent] = []
    current_indices: list[int] = []

    def flush() -> None:
        if not current_indices:
            return
        starts: list[float] = []
        ends: list[float] = []
        timestamps: list[float] = []
        ids: list[int] = []
        peak_index = current_indices[0]
        for index in current_indices:
            start, end = chunk_bounds(chunks[index])
            starts.append(float(start))
            ends.append(float(end))
            timestamps.append(float(chunk_timestamp(chunks[index])))
            ids.append(int(chunk_id(chunks[index])))
            if float(boundary_rows[index]["boundary_score"]) > float(boundary_rows[peak_index]["boundary_score"]):
                peak_index = index
        embeds = image_embeddings[current_indices]
        mean_embed = torch.nn.functional.normalize(embeds.mean(dim=0), dim=0).detach().cpu()
        representative_index = min(current_indices, key=lambda idx: abs(chunk_timestamp(chunks[idx]) - sum(timestamps) / len(timestamps)))
        events.append(
            LedgerEvent(
                event_id=len(events),
                start_time=min(starts),
                end_time=max(ends),
                member_chunk_ids=ids,
                member_timestamps=timestamps,
                mean_clip_embedding=mean_embed,
                peak_change_chunk_id=int(chunk_id(chunks[peak_index])),
                peak_change_score=float(boundary_rows[peak_index]["boundary_score"]),
                representative_chunk_id=int(chunk_id(chunks[representative_index])),
            )
        )

    for index, chunk in enumerate(chunks):
        start, end = chunk_bounds(chunk)
        should_start = False
        if current_indices:
            event_start = chunk_bounds(chunks[current_indices[0]])[0]
            duration = float(end) - float(event_start)
            if duration >= max_event_seconds:
                should_start = True
            if duration >= min_event_seconds and float(boundary_rows[index]["boundary_score"]) >= threshold:
                should_start = True
        if should_start:
            flush()
            current_indices = []
        current_indices.append(index)
    flush()
    return events


def parse_numeric_value(text: Any) -> int | None:
    raw = str(text).strip().lower()
    match = re.search(r"[-+]?\d+", raw)
    if match:
        return int(match.group(0))
    tokens = re.findall(r"[a-z]+", raw)
    for token in tokens:
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def numeric_options(options: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for option in options:
        value = parse_numeric_value(option.get("text", ""))
        if value is not None:
            out[str(option["letter"])] = int(value)
    return out


def option_for_count(count: int, options: list[dict[str, str]]) -> str | None:
    for letter, value in numeric_options(options).items():
        if int(value) == int(count):
            return letter
    return None


def semantic_count_query(question: str, numeric_option_mode: bool) -> str:
    text = re.sub(r"\s+", " ", str(question)).strip()
    if not numeric_option_mode:
        return text
    text = re.sub(r"\bhow many\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(number of|total number of|count of)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\?", "", text)
    return re.sub(r"\s+", " ", text).strip() or str(question).strip()


def score_events_for_question(
    scorer: EventLedgerClipScorer,
    events: list[LedgerEvent],
    question: str,
    options: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not events:
        return [], {"numeric_options": bool(numeric_options(options)), "queries": []}
    numeric = numeric_options(options)
    numeric_mode = bool(numeric) and len(numeric) == len(options)
    if numeric_mode:
        queries = [("?", semantic_count_query(question, numeric_option_mode=True))]
    elif options:
        queries = [(option["letter"], f"{question} {option['text']}".strip()) for option in options]
    else:
        queries = [("?", str(question).strip())]
    event_embeds = torch.stack([event.mean_clip_embedding for event in events], dim=0).to(scorer.device)
    text_embeds = scorer.text_embeddings([query for _letter, query in queries])
    scores = event_embeds @ text_embeds.T
    max_scores = torch.max(scores, dim=1)
    best_indices = max_scores.indices.detach().cpu().tolist()
    relevance = max_scores.values.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for event, best_index, score in zip(events, best_indices, relevance):
        rows.append(
            {
                "event_id": event.event_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "member_chunk_ids": event.member_chunk_ids,
                "member_timestamps": event.member_timestamps,
                "relevance": float(score),
                "best_supported_option": queries[int(best_index)][0],
                "peak_change_chunk_id": event.peak_change_chunk_id,
                "peak_change_score": event.peak_change_score,
                "representative_chunk_id": event.representative_chunk_id,
            }
        )
    return rows, {
        "numeric_options": bool(numeric_mode),
        "numeric_option_values": numeric,
        "semantic_count_query": semantic_count_query(question, numeric_option_mode=True) if numeric_mode else None,
        "queries": [{"letter": letter, "text": query} for letter, query in queries],
    }


def interval_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = float(left["start_time"]), float(left["end_time"])
    right_start, right_end = float(right["start_time"]), float(right["end_time"])
    inter = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return inter / union if union > 0 else 0.0


def deduplicate_events(events: list[dict[str, Any]], temporal_gap_seconds: float, iou_threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(events, key=lambda item: (float(item["start_time"]), float(item["end_time"])))
    clusters: list[dict[str, Any]] = []
    for event in ordered:
        if not clusters:
            clusters.append(
                {
                    "cluster_start": float(event["start_time"]),
                    "cluster_end": float(event["end_time"]),
                    "member_event_ids": [int(event["event_id"])],
                    "member_chunk_ids": list(event["member_chunk_ids"]),
                    "member_relevances": [float(event["relevance"])],
                    "max_relevance": float(event["relevance"]),
                    "mean_relevance": float(event["relevance"]),
                    "best_supported_option": event.get("best_supported_option"),
                }
            )
            continue
        last = clusters[-1]
        gap = float(event["start_time"]) - float(last["cluster_end"])
        pseudo_event = {"start_time": last["cluster_start"], "end_time": last["cluster_end"]}
        if gap <= temporal_gap_seconds or interval_iou(pseudo_event, event) >= iou_threshold:
            last["cluster_end"] = max(float(last["cluster_end"]), float(event["end_time"]))
            last["member_event_ids"].append(int(event["event_id"]))
            last["member_chunk_ids"].extend(int(value) for value in event["member_chunk_ids"])
            last["member_relevances"].append(float(event["relevance"]))
            relevances = last["member_relevances"]
            last["mean_relevance"] = float(sum(relevances) / len(relevances))
            if float(event["relevance"]) > float(last["max_relevance"]):
                last["max_relevance"] = float(event["relevance"])
                last["best_supported_option"] = event.get("best_supported_option")
        else:
            clusters.append(
                {
                    "cluster_start": float(event["start_time"]),
                    "cluster_end": float(event["end_time"]),
                    "member_event_ids": [int(event["event_id"])],
                    "member_chunk_ids": list(event["member_chunk_ids"]),
                    "member_relevances": [float(event["relevance"])],
                    "max_relevance": float(event["relevance"]),
                    "mean_relevance": float(event["relevance"]),
                    "best_supported_option": event.get("best_supported_option"),
                }
            )
    return clusters


def ledger_grid(
    event_scores: list[dict[str, Any]],
    options: list[dict[str, str]],
    gamma_values: list[float],
    delta_values: list[float],
    iou_threshold: float = 0.20,
) -> dict[str, dict[str, Any]]:
    grid: dict[str, dict[str, Any]] = {}
    for gamma in gamma_values:
        relevant = [event for event in event_scores if float(event["relevance"]) >= float(gamma)]
        for delta in delta_values:
            clusters = deduplicate_events(relevant, temporal_gap_seconds=float(delta), iou_threshold=float(iou_threshold))
            count = len(clusters)
            key = f"gamma{gamma:.2f}_delta{delta:.1f}"
            grid[key] = {
                "gamma_event": float(gamma),
                "dedup_delta_seconds": float(delta),
                "dedup_iou_threshold": float(iou_threshold),
                "relevant_event_ids": [int(event["event_id"]) for event in relevant],
                "clusters": clusters,
                "ledger_count": int(count),
                "mapped_option": option_for_count(count, options),
            }
    return grid


def event_to_json(event: LedgerEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "start_time": event.start_time,
        "end_time": event.end_time,
        "member_chunk_ids": event.member_chunk_ids,
        "member_timestamps": event.member_timestamps,
        "peak_change_chunk_id": event.peak_change_chunk_id,
        "peak_change_score": event.peak_change_score,
        "representative_chunk_id": event.representative_chunk_id,
    }
