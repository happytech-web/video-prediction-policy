#!/usr/bin/env python3
"""
Quick explorer for CALVIN datasets (incl. debug set).

Given a path to the dataset root or directly to a scene chunk (a directory
containing `scene_info.npy`), this script summarizes what is inside:
- discovers scene directories
- counts episodes and shows index ranges from `scene_info.npy`
- previews keys/shapes/dtypes from a few `episode_XXXXXXX.npz`
- previews language annotations from `lang_annotations/auto_lang_ann.npy`
  (ann text, index intervals, task ids if present)

Example:
  python scripts/explore_calvin_dataset.py /path/to/calvin_debug_dataset
  python scripts/explore_calvin_dataset.py /path/to/dataset/task_D_D/training
  python scripts/explore_calvin_dataset.py /path/to/dataset/task_D_D/validation
  python scripts/explore_calvin_dataset.py /path/to/dataset/task_D_D

Notes:
- Uses only stdlib + numpy. Optional: h5py if present (not required).
- Does not load images; only prints array shapes/dtypes for speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def is_scene_dir(p: Path) -> bool:
    return p.is_dir() and (p / "scene_info.npy").exists()


def discover_scene_dirs(root: Path, max_dirs: int = 200) -> List[Path]:
    """Find directories that contain `scene_info.npy`.

    If `root` itself is a scene dir, return [root]. Otherwise, search up to
    `max_dirs` matches recursively with a modest depth.
    """
    if is_scene_dir(root):
        return [root]

    matches: List[Path] = []
    # Limit breadth by stopping after accumulating `max_dirs`.
    for p in root.rglob("scene_info.npy"):
        scene_dir = p.parent
        matches.append(scene_dir)
        if len(matches) >= max_dirs:
            break
    return matches


def load_scene_indices(scene_dir: Path) -> Tuple[int, int, List[int]]:
    """Load index range from scene_info.npy.

    Returns: (start, end, full_index_list)
    """
    info = np.load(scene_dir / "scene_info.npy", allow_pickle=True).item()
    # According to CALVIN, `scene_info.npy` stores dict of {scene_name: (start_idx, end_idx)}
    # We pick the first entry.
    (start, end) = next(iter(info.values()))
    return int(start), int(end), list(range(int(start), int(end) + 1))


def list_episode_files(scene_dir: Path) -> List[Path]:
    return sorted(scene_dir.glob("episode_*.npz"))


def summarize_episode_npz(npz_path: Path) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Return a dict: key -> (shape, dtype) for arrays contained in an episode npz.
    Avoids loading image pixels fully; `np.load` already lazily exposes arrays.
    """
    out: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for k in data.files:
            try:
                arr = data[k]
                shape = tuple(arr.shape) if hasattr(arr, "shape") else ()
                dtype = str(arr.dtype) if hasattr(arr, "dtype") else type(arr).__name__
            except Exception as e:
                shape = ()
                dtype = f"error: {e}"  # best-effort reporting
            out[k] = (shape, dtype)
    return out


def preview_lang_annotations(scene_dir: Path, max_items: int = 5) -> Dict[str, object]:
    """Return basic info from lang_annotations/auto_lang_ann.npy if present.

    Extracts:
      - num_annotations
      - keys present under ['language'] and ['info']
      - sample entries: (start_idx, end_idx, ann, task?)
      - presence of 'task' or potential 'skill' fields
    """
    out: Dict[str, object] = {"exists": False}
    ann_path = scene_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        return out

    try:
        obj = np.load(ann_path, allow_pickle=True).item()
    except Exception as e:
        return {"exists": True, "error": f"failed to load: {e}"}

    out["exists"] = True
    out["language_keys"] = list(obj.get("language", {}).keys())
    out["info_keys"] = list(obj.get("info", {}).keys())

    # Some fields per CALVIN docs
    anns: List[str] = obj.get("language", {}).get("ann", [])
    tasks = obj.get("language", {}).get("task", None)
    idx_intervals: List[Tuple[int, int]] = obj.get("info", {}).get("indx", [])

    n = min(max_items, len(idx_intervals), len(anns) if isinstance(anns, list) else 0)
    samples: List[Dict[str, object]] = []
    for i in range(n):
        low, high = idx_intervals[i]
        item = {"range": (int(low), int(high)), "ann": anns[i] if isinstance(anns, list) else None}
        if isinstance(tasks, list) or isinstance(tasks, np.ndarray):
            try:
                item["task_id"] = int(tasks[i])
            except Exception:
                item["task_id"] = tasks[i]
        samples.append(item)

    out["num_annotations"] = len(idx_intervals)
    out["samples"] = samples

    # Heuristic: look for any key containing 'skill' (future-proofing user request)
    def scan_for_skill(d: object, path: str = "") -> List[str]:
        hits: List[str] = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_path = f"{path}/{k}" if path else k
                if "skill" in k.lower():
                    hits.append(new_path)
                hits.extend(scan_for_skill(v, new_path))
        return hits

    out["possible_skill_fields"] = scan_for_skill(obj)
    return out


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    p = argparse.ArgumentParser(description="Explore CALVIN dataset structure and samples")
    p.add_argument("path", type=str, help="Path to dataset root or a scene dir (contains scene_info.npy)")
    p.add_argument("--max-scenes", type=int, default=10, help="Max scene directories to summarize")
    p.add_argument("--max-episodes", type=int, default=3, help="Max episode files to peek per scene")
    p.add_argument("--max-lang", type=int, default=5, help="Max lang annotations to preview")
    args = p.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Path does not exist: {root}")
        return 1

    scenes = discover_scene_dirs(root)
    if not scenes:
        print("No scene directories found (no scene_info.npy discovered).")
        return 2

    print_header("Discovered Scene Directories")
    for i, s in enumerate(scenes[: args.max_scenes]):
        print(f"[{i:02d}] {s}")
    if len(scenes) > args.max_scenes:
        print(f"... and {len(scenes) - args.max_scenes} more")

    for i, scene_dir in enumerate(scenes[: args.max_scenes]):
        print_header(f"Scene {i:02d}: {scene_dir}")
        try:
            start, end, indices = load_scene_indices(scene_dir)
            print(f"Index range: {start}..{end} (total {len(indices)})")
        except Exception as e:
            print(f"Failed to read scene_info.npy: {e}")
            continue

        # Episodes
        episode_files = list_episode_files(scene_dir)
        print(f"Episodes (*.npz): {len(episode_files)} total")
        for ep in episode_files[: args.max_episodes]:
            print(f"- {ep.name}")
            info = summarize_episode_npz(ep)
            # Show a compact line per key: key: shape dtype
            for k, (shape, dtype) in sorted(info.items()):
                shape_str = "x".join(map(str, shape)) if shape else "-"
                print(f"  {k:20s}  shape=({shape_str:>10s})  dtype={dtype}")

        # Language annotations
        lang = preview_lang_annotations(scene_dir, max_items=args.max_lang)
        if not lang.get("exists", False):
            print("No lang_annotations/auto_lang_ann.npy found.")
        elif "error" in lang:
            print(f"Language annotations present but failed to load: {lang['error']}")
        else:
            print("Language annotations:")
            print(f"- keys(language): {lang.get('language_keys')}  keys(info): {lang.get('info_keys')}")
            print(f"- num_annotations: {lang.get('num_annotations')}  sample:{''}")
            for s in lang.get("samples", []):
                rng = s.get("range")
                ann = s.get("ann")
                tid = s.get("task_id", None)
                if tid is not None:
                    print(f"  [{rng[0]}..{rng[1]}]  task={tid}  ann=\"{ann}\"")
                else:
                    print(f"  [{rng[0]}..{rng[1]}]  ann=\"{ann}\"")
            skill_fields = lang.get("possible_skill_fields", [])
            if skill_fields:
                print(f"- possible 'skill' fields found at: {skill_fields}")
            else:
                print("- no 'skill' fields detected in annotation dict")

    print_header("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

