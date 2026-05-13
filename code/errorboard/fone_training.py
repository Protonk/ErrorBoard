"""Training loop for the FoNE arm.

Parallel to training.py but with:
  - FoneGPT model (input-side FoNE-features-add + dual output head)
  - StratifiedSamplerFone / EvalBatcherFone (emits num_values + sign/digit targets)
  - fone_loss + fone_correct (mixed vocab + per-digit CE; per-pair correctness)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .fone_dataset import (
    EvalBatcherFone,
    StratifiedSamplerFone,
    per_regime_eval_loaders_fone,
)
from .fone_model import FoneGPT, FoneGPTConfig, fone_correct, fone_loss
from .fone_tokenizer import (
    POS_NUM_C,
    POS_SIGN_C,
    SEQ_LEN,
    VOCAB_SIZE,
)
from . import preprocess as _add_preprocess
from . import preprocess_mult as _mul_preprocess
from . import preprocess_recip as _rec_preprocess
from . import regimes as _add_regimes
from . import mult_regimes as _mul_regimes
from . import recip_regimes as _rec_regimes
from .dataset import natural_distribution_weights


def _resolve_operation(name: str):
    if name == "add":
        return _add_preprocess, _add_regimes
    if name == "mul":
        return _mul_preprocess, _mul_regimes
    if name == "recip":
        return _rec_preprocess, _rec_regimes
    raise ValueError(f"unknown operation: {name!r}")


@dataclass
class FoneTrainingConfig:
    """Per-run FoNE training configuration."""

    run_name: str = "fone-default"
    runs_dir: str = "runs"
    seed: int = 0

    # operation: "add" (default) or "mul"
    operation: str = "add"

    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 48
    d_mlp: int = 192
    pos_encoding: str = "learned"
    init_std: float = 0.02

    # optimizer
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # schedule
    warmup_iters: int = 200
    max_iters: int = 20_000

    # data
    batch_size: int = 128
    holdout_frac: float = 0.10
    min_holdout: int = 10

    # eval / checkpoint
    eval_interval: int = 250
    eval_batch_size: int = 512
    train_eval_subsample: int = 256
    checkpoint_iters: tuple = (100, 250, 500, 1000, 2500, 5000, 10000, 20000)

    # device
    device: str = "cuda"

    # provenance
    git_commit: str = field(default="")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cosine_lr(iter_num: int, cfg: FoneTrainingConfig) -> float:
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * (iter_num + 1) / cfg.warmup_iters
    progress = (iter_num - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


_SIGN_POS = POS_SIGN_C - 1   # target-tensor index
_DIGIT_POS = POS_NUM_C - 1


@torch.no_grad()
def _eval_pool(model: FoneGPT, loader: EvalBatcherFone, device: torch.device,
               ) -> tuple[float, float, int]:
    """Iterate a loader and return (mean_total_loss, per-pair-accuracy, n_pairs).

    Loss is the FoNE total loss (vocab + digit). Accuracy is per-pair
    (sign + digit both correct, NaN handled).
    """
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_pairs = 0
    for batch in loader:
        input_ids = torch.from_numpy(batch["input_ids"]).long().to(device)
        target_tokens = torch.from_numpy(batch["target_tokens"]).long().to(device)
        num_values = torch.from_numpy(batch["num_values"]).float().to(device)
        sign_target = torch.from_numpy(batch["sign_target"]).long().to(device)
        digit_target = torch.from_numpy(batch["digit_target"]).long().to(device)
        is_nan = torch.from_numpy(batch["is_nan"]).to(device)

        out = model(input_ids, num_values)
        losses = fone_loss(out, target_tokens, sign_target, digit_target, is_nan)
        B = input_ids.shape[0]
        total_loss += float(losses["loss"].item()) * B
        correct = fone_correct(out, sign_target, digit_target, is_nan)
        n_correct += int(correct.sum().item())
        n_pairs += B
    model.train()
    if n_pairs == 0:
        return float("nan"), float("nan"), 0
    return total_loss / n_pairs, n_correct / n_pairs, n_pairs


def _train_subsample_loaders(
    table: np.ndarray,
    train_indices: np.ndarray,
    batch_size: int,
    n_per_regime: int,
    num_regimes: int,
) -> dict[int, EvalBatcherFone]:
    out: dict[int, EvalBatcherFone] = {}
    for r in range(num_regimes):
        mask = table["regime_id"][train_indices] == r
        idx = train_indices[mask][:n_per_regime]
        if len(idx) > 0:
            out[r] = EvalBatcherFone(table, idx, batch_size)
    return out


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "STATUS").write_text(status + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def train(cfg: FoneTrainingConfig) -> dict:
    """Run a single FoNE training session per cfg."""
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    cfg.git_commit = _git_commit()
    with open(run_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    (run_dir / "PID").write_text(f"{os.getpid()}\n")
    _write_status(run_dir, "running")

    try:
        _set_determinism(cfg.seed)
        device = torch.device(cfg.device)

        # ---- operation dispatch ----
        pp, rg = _resolve_operation(cfg.operation)
        regime_names = rg.REGIME_NAMES
        num_regimes = rg.NUM_REGIMES

        # ---- data ----
        table = pp.build_table()
        train_idx, holdout_idx = pp.split_train_holdout(
            table, holdout_frac=cfg.holdout_frac,
            min_holdout=cfg.min_holdout, seed=cfg.seed,
        )
        sampler = StratifiedSamplerFone(table, train_idx, seed=cfg.seed)
        holdout_eval_loaders = per_regime_eval_loaders_fone(
            table, holdout_idx, cfg.eval_batch_size,
        )
        train_eval_loaders = _train_subsample_loaders(
            table, train_idx, cfg.eval_batch_size, cfg.train_eval_subsample,
            num_regimes=num_regimes,
        )
        nat_weights = natural_distribution_weights(table)

        # ---- model ----
        fone_cfg = FoneGPTConfig(
            block_size=SEQ_LEN - 1,
            vocab_size=VOCAB_SIZE,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, d_mlp=cfg.d_mlp,
            pos_encoding=cfg.pos_encoding, init_std=cfg.init_std,
        )
        model = FoneGPT(fone_cfg).to(device)
        n_params = model.num_parameters()

        # ---- optimizer ----
        decay_params, nodecay_params = [], []
        for _, p in model.named_parameters():
            (decay_params if p.dim() >= 2 else nodecay_params).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=device.type == "cuda",
        )

        print(
            f"[{cfg.run_name}] model={n_params:,} params  pos_encoding={cfg.pos_encoding}  "
            f"tokenization=fone  vocab={VOCAB_SIZE}  seq_len={SEQ_LEN}  "
            f"device={device}  max_iters={cfg.max_iters}  batch={cfg.batch_size}",
            flush=True,
        )
        model.train()
        t_start = time.time()
        checkpoint_set = set(cfg.checkpoint_iters)

        for iter_num in range(cfg.max_iters + 1):
            # ---- eval + checkpoint ----
            do_eval = (iter_num % cfg.eval_interval == 0) or (iter_num in checkpoint_set)
            if do_eval:
                row: dict = {"iter": iter_num, "lr": _cosine_lr(iter_num, cfg)}

                holdout_loss: dict[str, float] = {}
                holdout_acc: dict[str, float] = {}
                for r, loader in holdout_eval_loaders.items():
                    if r not in sampler.active_regimes:
                        continue
                    lo, ac, _ = _eval_pool(model, loader, device)
                    holdout_loss[regime_names[r]] = lo
                    holdout_acc[regime_names[r]] = ac
                row["holdout_loss"] = holdout_loss
                row["holdout_acc"] = holdout_acc

                # Natural-distribution scalar.
                nat_num, nat_den = 0.0, 0.0
                for r in sampler.active_regimes:
                    name = regime_names[r]
                    if name in holdout_loss and not math.isnan(holdout_loss[name]):
                        w = float(nat_weights[r])
                        nat_num += w * holdout_loss[name]
                        nat_den += w
                row["natural_loss"] = nat_num / nat_den if nat_den > 0 else float("nan")

                train_loss: dict[str, float] = {}
                for r, loader in train_eval_loaders.items():
                    if r not in sampler.active_regimes:
                        continue
                    lo, _, _ = _eval_pool(model, loader, device)
                    train_loss[regime_names[r]] = lo
                row["train_loss"] = train_loss
                row["wall_time"] = time.time() - t_start

                _append_jsonl(metrics_path, row)
                print(
                    f"[iter {iter_num:6d}] nat_loss={row['natural_loss']:.4f}  "
                    f"lr={row['lr']:.2e}  t={row['wall_time']:.0f}s",
                    flush=True,
                )

                if iter_num in checkpoint_set:
                    torch.save(
                        {
                            "iter": iter_num,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "config": dataclasses.asdict(cfg),
                            "fone_gpt_config": dataclasses.asdict(fone_cfg),
                        },
                        run_dir / f"checkpoint_{iter_num:06d}.pt",
                    )

            if iter_num >= cfg.max_iters:
                break

            # ---- training step ----
            lr = _cosine_lr(iter_num, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = sampler.sample_batch(cfg.batch_size)
            input_ids = torch.from_numpy(batch["input_ids"]).long().to(device)
            target_tokens = torch.from_numpy(batch["target_tokens"]).long().to(device)
            num_values = torch.from_numpy(batch["num_values"]).float().to(device)
            sign_target = torch.from_numpy(batch["sign_target"]).long().to(device)
            digit_target = torch.from_numpy(batch["digit_target"]).long().to(device)
            is_nan = torch.from_numpy(batch["is_nan"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids, num_values)
            losses = fone_loss(
                out, target_tokens, sign_target, digit_target, is_nan,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        _write_status(run_dir, "completed")
        print(f"[{cfg.run_name}] completed in {time.time() - t_start:.0f}s", flush=True)
        return {"status": "completed", "final_iter": cfg.max_iters, "n_params": n_params}

    except KeyboardInterrupt:
        _write_status(run_dir, "killed")
        raise
    except Exception:
        _write_status(run_dir, "failed")
        traceback.print_exc()
        raise


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train a FoNE-arm mouse model.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=48)
    p.add_argument("--d-mlp", type=int, default=192)
    p.add_argument("--pos-encoding", default="learned", choices=["learned", "rope"])
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = FoneTrainingConfig(
        run_name=args.run_name, runs_dir=args.runs_dir, seed=args.seed,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, d_mlp=args.d_mlp,
        pos_encoding=args.pos_encoding,
        max_iters=args.max_iters, warmup_iters=args.warmup_iters,
        batch_size=args.batch_size, learning_rate=args.learning_rate, min_lr=args.min_lr,
        eval_interval=args.eval_interval, device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
