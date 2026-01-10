#!/usr/bin/env python3
"""
Extract unique task names from CALVIN dataset language annotations.

Given a dataset root (e.g., CALVIN debug set, or a split folder like
task_D_D/training), the script:
- discovers scene directories (contain `scene_info.npy`)
- loads `lang_annotations/auto_lang_ann.npy` if present
- aggregates unique `language.task` entries and counts

Usage:
  python scripts/extract_calvin_tasks.py /path/to/calvin_debug_dataset
  python scripts/extract_calvin_tasks.py /path/to/dataset/task_D_D/training

Options:
  --max-scenes N    Limit number of scene dirs to scan (default: all)
  --save PATH       Save a JSON summary with tasks, counts, and per-scene stats

Only numpy and stdlib are required.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def is_scene_dir(p: Path) -> bool:
    return p.is_dir() and (p / "scene_info.npy").exists()


def discover_scene_dirs(root: Path) -> List[Path]:
    if is_scene_dir(root):
        return [root]
    return [p.parent for p in root.rglob("scene_info.npy")]


def load_lang_ann(scene_dir: Path):
    ann_path = scene_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        return None
    try:
        return np.load(ann_path, allow_pickle=True).item()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract unique tasks from CALVIN dataset")
    ap.add_argument("path", type=str, help="Dataset root or scene directory")
    ap.add_argument("--max-scenes", type=int, default=None, help="Limit number of scene dirs to scan")
    ap.add_argument("--save", type=str, default=None, help="Save JSON summary to this path")
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

    global_counter: Counter = Counter()
    per_scene_counts: Dict[str, Dict[str, int]] = {}
    per_scene_num_anns: Dict[str, int] = {}

    for scene_dir in scenes:
        data = load_lang_ann(scene_dir)
        if data is None:
            continue
        tasks = data.get("language", {}).get("task", [])
        # tasks can be list of strings or numeric ids depending on split/version
        # We convert to strings for a unified view, but keep raw in JSON if saved.
        scene_counter = Counter(map(lambda x: str(x), list(tasks)))
        global_counter.update(scene_counter)
        per_scene_counts[str(scene_dir)] = dict(scene_counter)
        idx_intervals = data.get("info", {}).get("indx", [])
        per_scene_num_anns[str(scene_dir)] = len(idx_intervals)

    # Print summary
    unique_tasks = sorted(global_counter.keys())
    print("=" * 80)
    print(f"Scanned scenes: {len(per_scene_counts)} / {len(scenes)}")
    print(f"Unique tasks: {len(unique_tasks)}")
    print("- Tasks (sorted):")
    for t in unique_tasks:
        print(f"  {t}")
    print("- Counts (global):")
    for t, c in global_counter.most_common():
        print(f"  {t}: {c}")

    if args.save:
        summary = {
            "scenes_scanned": len(per_scene_counts),
            "unique_tasks": unique_tasks,
            "global_counts": dict(global_counter),
            "per_scene_counts": per_scene_counts,
            "per_scene_num_annotations": per_scene_num_anns,
        }
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

