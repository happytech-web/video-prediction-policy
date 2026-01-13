import random
from collections import deque, defaultdict
from typing import Dict, Iterable, Iterator, List, Optional

import torch


class GroupedNoReplacementBatchSampler:
    """
    BatchSampler that groups dataset indices by a provided skill_id -> indices mapping and
    yields batches without replacement. It aims to include at least `min_per_skill` samples
    for each selected skill in a batch when available. If a skill has fewer than
    `min_per_skill` remaining samples, it will take whatever remains. Any remaining slots
    in the batch are filled from the global pool without additional constraints.

    - No replacement within an epoch.
    - Supports distributed training by emitting only the subset of batches assigned to this rank.

    Args:
        buckets: Mapping from skill_id to list of dataset indices (global, 0..len-1).
        batch_size: Target batch size.
        min_per_skill: Minimum number of samples per selected skill to try to include.
        max_skills_per_batch: Optional cap on number of distinct skills per batch. If None,
            it is derived as batch_size // min_per_skill.
        drop_last: Whether to drop the final incomplete batch.
        seed: Base seed for shuffling per epoch.
    """

    def __init__(
        self,
        buckets: Dict[int, List[int]],
        batch_size: int,
        min_per_skill: int = 2,
        max_skills_per_batch: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 1234,
    ) -> None:
        assert batch_size > 0
        assert min_per_skill >= 1
        self._orig_buckets = {k: list(v) for k, v in buckets.items() if len(v) > 0}
        self.batch_size = int(batch_size)
        self.min_per_skill = int(min_per_skill)
        self.max_skills_per_batch = (
            int(max_skills_per_batch) if max_skills_per_batch is not None else None
        )
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        # distributed info (optional)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = sum(len(v) for v in self._orig_buckets.values())
        if self.drop_last:
            full = total // self.batch_size
        else:
            full = (total + self.batch_size - 1) // self.batch_size
        # Return local length per-rank
        return (full + self.world_size - 1) // self.world_size

    def __iter__(self) -> Iterator[List[int]]:
        # Create local copies and shuffle deterministically for this epoch
        rnd = random.Random(self.seed + self.epoch)
        buckets: Dict[int, deque] = {}
        for k, idxs in self._orig_buckets.items():
            ii = list(idxs)
            rnd.shuffle(ii)
            buckets[k] = deque(ii)

        skills = list(buckets.keys())
        rnd.shuffle(skills)

        # helper: count remaining
        def remaining_skills() -> List[int]:
            return [k for k in skills if len(buckets[k]) > 0]

        # global batch index for rank slicing
        global_batch_idx = 0
        while True:
            active = remaining_skills()
            if not active:
                break

            # derive K (skills per batch)
            if self.max_skills_per_batch is not None:
                K = min(self.max_skills_per_batch, len(active))
            else:
                K = max(1, min(len(active), self.batch_size // self.min_per_skill))

            # prioritize skills with >= min_per_skill remaining
            rich = [k for k in active if len(buckets[k]) >= self.min_per_skill]
            poor = [k for k in active if len(buckets[k]) < self.min_per_skill]
            rnd.shuffle(rich)
            rnd.shuffle(poor)
            chosen: List[int] = (rich + poor)[:K]

            batch: List[int] = []
            # take up to min_per_skill from chosen skills without replacement
            for k in chosen:
                take = min(self.min_per_skill, len(buckets[k]))
                for _ in range(take):
                    batch.append(buckets[k].popleft())

            # fill the remainder freely from any remaining indices (no replacement)
            if len(batch) < self.batch_size:
                # gather a round-robin list of remaining indices
                fill_needed = self.batch_size - len(batch)
                # simple round-robin over all skills still having data
                fill_skills = remaining_skills()
                fill_ptr = 0
                while fill_needed > 0 and fill_skills:
                    k = fill_skills[fill_ptr]
                    if buckets[k]:
                        batch.append(buckets[k].popleft())
                        fill_needed -= 1
                        if not buckets[k]:
                            # drop exhausted skill for future rounds
                            fill_skills.pop(fill_ptr)
                            if not fill_skills:
                                break
                            fill_ptr %= len(fill_skills) if fill_skills else 1
                            continue
                    fill_ptr = (fill_ptr + 1) % len(fill_skills) if fill_skills else 0

            # if batch is empty, stop
            if not batch:
                break

            # apply drop_last policy
            if len(batch) < self.batch_size and self.drop_last:
                # if dropping last incomplete batch, terminate
                break

            # distributed slicing: only yield batches assigned to this rank
            if (global_batch_idx % self.world_size) == self.rank:
                yield batch
            global_batch_idx += 1


def build_skill_buckets_from_dataset(dataset) -> Dict[int, List[int]]:
    """
    Build skill buckets using dataset's episode lookup and task-id lookup.
    Requires the dataset to expose `episode_lookup` and `_lookup_task_id(start_idx)`.
    Returns a dict: skill_id -> list of dataset indices.
    """
    buckets: Dict[int, List[int]] = defaultdict(list)
    if not hasattr(dataset, "episode_lookup") or not hasattr(dataset, "_lookup_task_id"):
        return {}
    ep_lookup = dataset.episode_lookup
    n = len(ep_lookup)
    for i in range(n):
        try:
            start_idx = int(ep_lookup[i])
            tid = dataset._lookup_task_id(start_idx)
            if tid is None:
                continue
            tid = int(tid)
            buckets[tid].append(i)
        except Exception:
            continue
    # remove empty buckets
    buckets = {k: v for k, v in buckets.items() if len(v) > 0}
    return buckets

