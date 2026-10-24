"""Training loop, prediction, and validation: PQ of the predicted instances (at full resolution)
against every annotation set of each validation observation, counts pooled over the dataset."""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, get_worker_info

from .metrics import match_instances, pq_from_counts
from .postprocess import components, drop_small, upsample
from .preprocess import disk_mask


def seed_worker(_):
    info = get_worker_info()
    info.dataset.reseed(info.seed % 2 ** 31)


def make_loader(ds, batch_size: int, train: bool, workers: int = 4) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=workers,
                      worker_init_fn=seed_worker, pin_memory=torch.cuda.is_available())


def fit(model, loader, loss_fn, epochs: int, lr: float = 3e-4, weight_decay: float = 1e-4,
        warmup_epochs: float = 1.0, eval_fn=None, eval_every: int = 2, device=None, log=print) -> pd.DataFrame:
    """AdamW, linear warmup then cosine decay, fp16 autocast on GPU. If eval_fn(model) -> score is given,
    the weights of the best-scoring evaluation are restored at the end. Returns the per-epoch history."""
    device = device or next(model.parameters()).device
    amp = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total, warm = epochs * len(loader), max(1, int(warmup_epochs * len(loader)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=amp)
    best, best_state, history, t0 = -1.0, None, [], time.time()

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": ep, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0]}
        if eval_fn is not None and (ep % eval_every == 0 or ep == epochs):
            rec["val_pq"] = eval_fn(model)
            if rec["val_pq"] > best:
                best, best_state = rec["val_pq"], copy.deepcopy(model.state_dict())
        rec["elapsed_s"] = time.time() - t0
        history.append(rec)
        log(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


@torch.no_grad()
def predict(model, loader, device=None) -> dict:
    """{obs: filament probability (float16) at working resolution}."""
    device = device or next(model.parameters()).device
    model.eval()
    out = {}
    for x, _, idx in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        for i, p in zip(idx.tolist(), torch.sigmoid(logits.float())[:, 0].cpu().numpy()):
            out[loader.dataset.obs[i]] = p.astype(np.float16)
    return out


def evaluate(probs: dict, gt_fn, disks: dict, thresholds=(0.5,), min_areas=(0,), size: int = 2048,
             limb_cut: float = 0.985) -> pd.DataFrame:
    """Per-image TP / FP / FN / IoU sums for every (threshold, min_area), each summed over the
    observation's annotation sets. gt_fn(obs) -> list of full-resolution instance maps;
    disks: {obs: full-resolution (cx, cy, radius)}. Pool with `summarise`."""
    rows = []
    for o, prob in probs.items():
        gts = gt_fn(o)
        p = upsample(prob, size)
        on_disk = disk_mask(p.shape, *disks[o], limb_cut)
        for t in thresholds:
            lab, areas = components((p > t) & on_disk)
            for a in min_areas:
                pred = drop_small(lab, areas, a)
                c = np.zeros(4)
                for gt in gts:
                    m = match_instances(gt, pred)
                    c += (m["tp"], m["fp"], m["fn"], m["iou_sum"])
                rows.append({"obs": o, "threshold": t, "min_area": a, "n_sets": len(gts),
                             "tp": c[0], "fp": c[1], "fn": c[2], "iou_sum": c[3]})
    return pd.DataFrame(rows)


def summarise(counts: pd.DataFrame, by=("threshold", "min_area")) -> pd.DataFrame:
    """Pooled PQ / SQ / RQ per group of `evaluate` rows (by=["obs", ...] gives per-image scores)."""
    sums = counts.groupby(list(by))[["tp", "fp", "fn", "iou_sum"]].sum()
    return pd.DataFrame([pq_from_counts(int(r.tp), int(r.fp), int(r.fn), r.iou_sum) for r in sums.itertuples()],
                        index=sums.index)
