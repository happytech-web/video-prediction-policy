#!/usr/bin/env python3
"""
Build per-annotation and per-frame index for CALVIN datasets using task-as-label.

What it does
- Scans one dataset root (containing `training` and/or `validation`) or a single
  scene directory (containing `scene_info.npy`).
- Loads language annotations from `lang_annotations/auto_lang_ann.npy` (configurable
  via --lang-folder) in each scene.
- Collects unique tasks and assigns stable integer ids (task-as-skill).
- Emits JSON files with:
  - tasks mapping: task -> id
  - per-scene annotation windows with task, task_id, and text
  - optional per-frame labels for frames falling into any annotated window

Usage
  python scripts/build_calvin_task_index.py /path/to/calvin_debug_dataset \
      --out ./calvin_task_index_debug.json

  python scripts/build_calvin_task_index.py /path/to/dataset/task_D_D/training \
      --out ./calvin_task_index_train.json

Options
  --lang-folder NAME   Folder name holding auto_lang_ann.npy (default: lang_annotations)
  --include-frames     Include expanded per-frame labels (may be large)
  --max-scenes N       Limit scenes to speed up (default: all)

The output JSON is self-contained and can be used by a future dataloader to
sample positives/negatives by task id without re-parsing the dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Annotation:
    ann_id: int
    start: int
    end: int
    task: str
    task_id: int
    text: Optional[str]


@dataclass
class SceneSummary:
    path: str
    split: Optional[str]
    index_start: int
    index_end: int
    num_episodes: int
    annotations: List[Annotation]


def is_scene_dir(p: Path) -> bool:
    return p.is_dir() and (p / "scene_info.npy").exists()


def discover_scene_dirs(root: Path) -> List[Path]:
    if is_scene_dir(root):
        return [root]
    return [p.parent for p in root.rglob("scene_info.npy")]


def load_scene_indices(scene_dir: Path) -> Tuple[int, int, List[int]]:
    info = np.load(scene_dir / "scene_info.npy", allow_pickle=True).item()
    (start, end) = next(iter(info.values()))
    return int(start), int(end), list(range(int(start), int(end) + 1))


def list_episode_files(scene_dir: Path) -> List[Path]:
    return sorted(scene_dir.glob("episode_*.npz"))


def load_lang_data(scene_dir: Path, lang_folder: str) -> Optional[dict]:
    # Prefer configured lang folder; fall back to root if needed
    cand = [scene_dir / lang_folder / "auto_lang_ann.npy", scene_dir / "auto_lang_ann.npy"]
    for p in cand:
        if p.exists():
            try:
                return np.load(p, allow_pickle=True).item()
            except Exception:
                continue
    return None


def infer_split(path: Path) -> Optional[str]:
    s = path.as_posix()
    if "/training" in s:
        return "training"
    if "/validation" in s:
        return "validation"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CALVIN task index (task-as-label)")
    ap.add_argument("path", type=str, help="Dataset root or scene directory")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path")
    ap.add_argument("--lang-folder", type=str, default="lang_annotations", help="Folder containing auto_lang_ann.npy")
    ap.add_argument("--include-frames", action="store_true", help="Include expanded per-frame labels (may be large)")
    ap.add_argument("--max-scenes", type=int, default=None, help="Limit scenes scanned")
    args = ap.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Path does not exist: {root}")
        return 1

    scenes = discover_scene_dirs(root)
    if not scenes:
        print("No scene directories found (scene_info.npy not discovered).")
        return 2
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    # First pass: gather all unique tasks
    tasks_set: set[str] = set()
    lang_per_scene: Dict[Path, dict] = {}
    for sdir in scenes:
        lang = load_lang_data(sdir, args.lang_folder)
        if lang is None:
            continue
        lang_per_scene[sdir] = lang
        tasks = lang.get("language", {}).get("task", [])
        for t in tasks:
            tasks_set.add(str(t))

    tasks_sorted = sorted(tasks_set)
    task_to_id: Dict[str, int] = {t: i for i, t in enumerate(tasks_sorted)}

    # Second pass: build summaries and optional per-frame labels
    scene_summaries: List[SceneSummary] = []
    frame_labels: List[Dict[str, object]] = []

    for sdir in scenes:
        if sdir not in lang_per_scene:
            continue
        split = infer_split(sdir)
        start_idx, end_idx, _ = load_scene_indices(sdir)
        num_eps = len(list_episode_files(sdir))

        lang = lang_per_scene[sdir]
        idx_intervals: List[Tuple[int, int]] = lang.get("info", {}).get("indx", [])
        tasks = list(map(str, lang.get("language", {}).get("task", [])))
        texts = lang.get("language", {}).get("ann", [])

        anns: List[Annotation] = []
        for i, (low, high) in enumerate(idx_intervals):
            task = tasks[i] if i < len(tasks) else ""
            text = texts[i] if i < len(texts) else None
            anns.append(
                Annotation(
                    ann_id=i,
                    start=int(low),
                    end=int(high),
                    task=task,
                    task_id=task_to_id.get(task, -1),
                    text=text,
                )
            )

        scene_summaries.append(
            SceneSummary(
                path=sdir.as_posix(),
                split=split,
                index_start=start_idx,
                index_end=end_idx,
                num_episodes=num_eps,
                annotations=anns,
            )
        )

        if args.include_frames:
            # Expand to per-frame labels for annotated frames only
            for a in anns:
                for f in range(a.start, a.end + 1):
                    frame_labels.append(
                        {
                            "scene": sdir.as_posix(),
                            "frame": f,
                            "ann_id": a.ann_id,
                            "task": a.task,
                            "task_id": a.task_id,
                        }
                    )

    out = {
        "root": root.as_posix(),
        "lang_folder": args.lang_folder,
        "tasks": task_to_id,
        "num_scenes": len(scene_summaries),
        "scenes": [
            {
                **{k: v for k, v in asdict(s).items() if k != "annotations"},
                "annotations": [asdict(a) for a in s.annotations],
            }
            for s in scene_summaries
        ],
    }
    if args.include_frames:
        out["frames"] = frame_labels

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote index to: {out_path}")
    print(f"Unique tasks: {len(task_to_id)}  Scenes: {len(scene_summaries)}")
    if args.include_frames:
        print(f"Per-frame labels: {len(frame_labels)} entries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

