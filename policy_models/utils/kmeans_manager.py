import math
import numpy as np
from typing import Optional, Tuple, List
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _gather_cat_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    All-gather variable-length local tensor to all ranks and concatenate.
    Works for 1D or 2D tensors; gathers along dim 0.
    """
    if not dist.is_available() or not dist.is_initialized():
        return t
    world_size = dist.get_world_size()
    device = t.device
    local_len = torch.tensor([t.shape[0]], device=device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    lens = torch.stack(lens).cpu().tolist()
    max_len = int(max(l[0] for l in lens))
    pad = max_len - t.shape[0]
    if pad > 0:
        if t.ndim == 1:
            pad_t = torch.zeros((pad,), device=device, dtype=t.dtype)
        else:
            pad_shape = (pad,) + tuple(t.shape[1:])
            pad_t = torch.zeros(pad_shape, device=device, dtype=t.dtype)
        t_pad = torch.cat([t, pad_t], dim=0)
    else:
        t_pad = t
    gather_list = [torch.zeros_like(t_pad) for _ in range(world_size)]
    dist.all_gather(gather_list, t_pad)
    chunks: List[torch.Tensor] = []
    for gi, ln in zip(gather_list, lens):
        chunks.append(gi[: ln[0]])
    return torch.cat(chunks, dim=0)


class KMeansManager:
    """
    Handles alternating full-dataset feature extraction and KMeans clustering.
    - Multi-GPU: each rank extracts its shard; rank0 runs KMeans (faiss GPU if available),
      then broadcasts centers and assignments to all ranks.
    - Fallback: if faiss unavailable, uses simple k-means in torch (CPU) as a fallback.
    """

    def __init__(self, n_clusters: int, feature_dim: int, pca_dim: int = 256):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.pca_dim = pca_dim
        self.centers_mu: Optional[torch.Tensor] = None  # [K,D]
        self.assignments: Optional[torch.Tensor] = None  # [N]

    @torch.no_grad()
    def extract_features(
        self,
        model,
        dataloader,
        device: torch.device,
        extract_fn,  # callable: batch -> (h: [B,D], idx: [B])
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns concatenated features [N,D] and indices [N] across all ranks.
        """
        feats_list: List[torch.Tensor] = []
        idx_list: List[torch.Tensor] = []
        model.eval()
        for batch in dataloader:
            h, idx = extract_fn(batch)
            if h is None or idx is None:
                continue
            feats_list.append(h.to(device))
            idx_list.append(idx.to(device).long())
        if not feats_list:
            return torch.empty(0, self.feature_dim, device=device), torch.empty(0, dtype=torch.long, device=device)
        feats_local = torch.cat(feats_list, dim=0)
        idx_local = torch.cat(idx_list, dim=0)
        feats_all = _gather_cat_tensor(feats_local)
        idx_all = _gather_cat_tensor(idx_local)
        return feats_all, idx_all

    def _preprocess_features_faiss(self, x_np: np.ndarray) -> np.ndarray:
        """
        Mirror SPCL preprocessing: PCA (whiten) to pca_dim, then L2 normalize.
        """
        faiss = _import_faiss()
        n, d = x_np.shape
        x_np = x_np.astype("float32")
        out_dim = min(self.pca_dim, d)
        if out_dim > 0 and out_dim < d:
            mat = faiss.PCAMatrix(d, out_dim, eigen_power=-0.5)
            mat.train(x_np)
            assert mat.is_trained
            x_np = mat.apply_py(x_np)
        # L2 normalize rows
        norms = np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-12
        x_np = x_np / norms
        return x_np

    def _kmeans_faiss(self, x_np: np.ndarray) -> np.ndarray:
        faiss = _import_faiss()
        if faiss is None:
            raise ImportError("faiss not available")
        x_np = self._preprocess_features_faiss(x_np)
        n, d = x_np.shape
        clus = faiss.Clustering(d, self.n_clusters)
        clus.seed = np.random.randint(1234)
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        # Prefer GPU if available, else CPU index
        try:
            ngpu = faiss.get_num_gpus()
        except Exception:
            ngpu = 0
        if ngpu and ngpu > 0:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False
            flat_config.device = 0
            index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            index = faiss.IndexFlatL2(d)
        clus.train(x_np, index)
        _, I = index.search(x_np, 1)
        return I.astype(np.int64).reshape(-1)


    def _compute_centers_from_assign(self, x: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
        K = self.n_clusters
        D = x.shape[1]
        centers = torch.zeros(K, D, dtype=x.dtype, device=x.device)
        for k in range(K):
            mask = assign == k
            if mask.any():
                centers[k] = x[mask].mean(dim=0)
        return centers

    def _kmeans_torch(self, x: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
        """Simple K-Means in torch as a fallback. Returns assignments [N]."""
        device = x.device
        N = x.shape[0]
        K = self.n_clusters
        if N < K:
            # degenerate; assign all to zero cluster
            return torch.zeros(N, dtype=torch.long, device=device)
        # init centers by random samples
        perm = torch.randperm(N, device=device)
        centers = x[perm[:K]].clone()
        for _ in range(n_iter):
            # assign
            # dist^2 = |x|^2 + |c|^2 - 2 x c
            x2 = (x * x).sum(dim=1, keepdim=True)  # [N,1]
            c2 = (centers * centers).sum(dim=1).unsqueeze(0)  # [1,K]
            sim = x @ centers.t()  # [N,K]
            d2 = x2 + c2 - 2.0 * sim
            assign = d2.argmin(dim=1)
            # update
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    new_centers[k] = x[mask].mean(dim=0)
                else:
                    # re-seed empty cluster
                    ridx = torch.randint(0, N, (1,), device=device)
                    new_centers[k] = x[ridx]
            if torch.allclose(new_centers, centers, atol=1e-4, rtol=0.0):
                centers = new_centers
                break
            centers = new_centers
        return assign

    def run_kmeans(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run kmeans on rank0, return (assignments[N], centers[K,D])."""
        # normalize to unit length (pre-FAISS PCA+L2 also normalizes internally)
        x_n = torch.nn.functional.normalize(x, dim=1)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            try:
                faiss = _import_faiss()
                if faiss is None:
                    raise ImportError("faiss unavailable")
                assign_np = self._kmeans_faiss(x_n.cpu().numpy())
                assign = torch.from_numpy(assign_np).long()
                centers = self._compute_centers_from_assign(x_n, assign).to(x.dtype)
                logger.info("KMeans: used FAISS backend")
            except Exception as e:
                logger.warning(f"KMeans: FAISS backend unavailable ({e}); falling back to torch implementation")
                assign = self._kmeans_torch(x_n).long()
                centers = self._compute_centers_from_assign(x_n, assign).to(x.dtype)
        else:
            assign = torch.empty(x_n.shape[0], dtype=torch.long)
            centers = torch.empty(self.n_clusters, x_n.shape[1], dtype=x.dtype)

        # broadcast results to all ranks
        if dist.is_available() and dist.is_initialized():
            device = x.device
            assign = assign.to(device)
            centers = centers.to(device)
            # sizes:
            n = torch.tensor([assign.numel()], device=device, dtype=torch.long)
            dist.broadcast(n, src=0)
            # pad assign to n across ranks if needed
            if dist.get_rank() != 0:
                assign = torch.empty(n.item(), dtype=torch.long, device=device)
            dist.broadcast(assign, src=0)
            dist.broadcast(centers, src=0)

        return assign, centers

    def update(self, features: torch.Tensor, indices: torch.Tensor):
        """
        Update internal state given full-dataset features and corresponding dataset indices.
        """
        assign, centers = self.run_kmeans(features)
        # reorder to align by dataset index
        # features/indices are in arbitrary order; we build a global assignment array by max index+1
        N_total = int(indices.max().item()) + 1
        device = features.device
        full_assign = torch.empty(N_total, dtype=torch.long, device=device)
        full_assign[indices] = assign
        self.assignments = full_assign.detach()
        self.centers_mu = centers.detach()

    def ready(self) -> bool:
        return self.assignments is not None and self.centers_mu is not None

    def get_centers(self) -> Optional[torch.Tensor]:
        return self.centers_mu

    def get_assignments(self, idx: torch.Tensor) -> Optional[torch.Tensor]:
        if self.assignments is None:
            return None
        return self.assignments[idx]
