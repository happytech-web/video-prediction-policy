import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    Encapsulates projection head g_c, metric head g_m, and contrastive/prototype loss utilities.

    Exposes per-loss methods so the policy can decide what to compute.
    """

    def __init__(
        self,
        dim: int,
        gm_hidden: int = 256,
        tau_contra: float = 0.07,
        tau_proto: float = 0.1,
    ) -> None:
        super().__init__()
        self.tau_contra = tau_contra
        self.tau_proto = tau_proto

        # g_c: projection head for contrastive z = g_c(h)
        self.gc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # g_m: small MLP to turn distances into binary logits
        self.gm = nn.Sequential(
            nn.Linear(1, gm_hidden),
            nn.GELU(),
            nn.LayerNorm(gm_hidden),
            nn.Linear(gm_hidden, gm_hidden),
            nn.GELU(),
            nn.LayerNorm(gm_hidden),
            nn.Linear(gm_hidden, 2),
        )

    # ---------- helpers ----------
    def z(self, h: torch.Tensor) -> torch.Tensor:
        return self.gc(h)

    # ---------- losses ----------
    def loss_spcl(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        temperature: float = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        SPCL (SimCLR-style) NT-Xent loss with two views per sample (no labels required).
        Matches ref/SPCL/model/losses.py::SPCLCriterion.
        """
        T = self.tau_contra if temperature is None else temperature
        if normalize:
            z_i = F.normalize(z_i, dim=-1)
            z_j = F.normalize(z_j, dim=-1)

        bsz = z_i.shape[0]
        device = z_i.device

        logits_aa = (z_i @ z_i.t()) / T
        logits_bb = (z_j @ z_j.t()) / T
        logits_ab = (z_i @ z_j.t()) / T
        logits_ba = (z_j @ z_i.t()) / T

        mask = torch.ones((bsz, bsz), dtype=torch.bool, device=device).fill_diagonal_(False)

        pos_ab = logits_ab[~mask]  # [bsz]
        pos_ba = logits_ba[~mask]  # [bsz]
        pos = torch.cat((pos_ab, pos_ba), dim=0).unsqueeze(1)  # [2*bsz,1]

        neg_aa = logits_aa[mask].reshape(bsz, -1)
        neg_bb = logits_bb[mask].reshape(bsz, -1)
        neg_ab = logits_ab[mask].reshape(bsz, -1)
        neg_ba = logits_ba[mask].reshape(bsz, -1)

        neg_a = torch.cat((neg_aa, neg_ab), dim=1)
        neg_b = torch.cat((neg_ba, neg_bb), dim=1)
        neg = torch.cat((neg_a, neg_b), dim=0)  # [2*bsz, 2*(bsz-1)]

        logits = torch.cat((pos, neg), dim=1)  # [2*bsz, 2*(bsz-1)+1]
        labels = torch.zeros(2 * bsz, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    def loss_contra(self, h: torch.Tensor, skill_id: torch.LongTensor, tau: float = None) -> torch.Tensor:
        """NT-Xent contrastive loss treating same skill as positive."""
        tau = self.tau_contra if tau is None else tau
        z = F.normalize(self.z(h), dim=-1)
        B = z.shape[0]
        sim = (z @ z.t()) / tau
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(B, device=z.device, dtype=torch.bool)
        pos_mask = (skill_id[:, None] == skill_id[None, :]) & (~eye)
        exp_sim = sim.exp()
        denom = exp_sim.masked_fill(eye, 0.0).sum(dim=1)
        log_prob = sim - torch.log(denom.unsqueeze(1) + 1e-8)
        pos_cnt = pos_mask.sum(dim=1)
        if not (pos_cnt > 0).any():
            return torch.tensor(0.0, device=z.device)
        pos_log_sum = (log_prob * pos_mask.float()).sum(dim=1)
        loss_per_i = -pos_log_sum / (pos_cnt + 1e-8)
        return loss_per_i.mean()

    def loss_proto(
        self,
        h: torch.Tensor,
        centers_mu: torch.Tensor,
        cluster_id: torch.LongTensor,
        tau: float = None,
    ) -> torch.Tensor:
        """Prototypical cross-entropy against KMeans labels c(i)."""
        tau = self.tau_proto if tau is None else tau
        h_n = F.normalize(h, dim=-1)
        mu_n = F.normalize(centers_mu, dim=-1)
        logits = (h_n @ mu_n.t()) / tau
        return F.cross_entropy(logits, cluster_id.long())

    def loss_metric(
        self,
        h: torch.Tensor,
        cluster_id: torch.LongTensor,
        max_pairs: int = 8192,
    ) -> torch.Tensor:
        """Prototype-level siamese metric loss using g_m on pairwise distances."""
        device = h.device
        B = h.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)
        idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=device)
        same = cluster_id[idx_i] == cluster_id[idx_j]
        diff = ~same
        pos_i = idx_i[same]
        pos_j = idx_j[same]
        neg_i = idx_i[diff]
        neg_j = idx_j[diff]

        def _sample_pairs(pi, pj):
            n = pi.numel()
            if n <= max_pairs:
                return pi, pj
            perm = torch.randperm(n, device=device)[:max_pairs]
            return pi[perm], pj[perm]

        pos_i, pos_j = _sample_pairs(pos_i, pos_j)
        neg_i, neg_j = _sample_pairs(neg_i, neg_j)

        loss_terms = []
        if pos_i.numel() > 0:
            d_pos = (h[pos_i] - h[pos_j]).norm(dim=-1, keepdim=True)
            logit_pos = self.gm(d_pos)
            y_pos = torch.ones(logit_pos.shape[0], device=device, dtype=torch.long)
            loss_terms.append(F.cross_entropy(logit_pos, y_pos))

        if neg_i.numel() > 0:
            d_neg = (h[neg_i] - h[neg_j]).norm(dim=-1, keepdim=True)
            logit_neg = self.gm(d_neg)
            y_neg = torch.zeros(logit_neg.shape[0], device=device, dtype=torch.long)
            loss_terms.append(F.cross_entropy(logit_neg, y_neg))

        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(loss_terms).mean()
