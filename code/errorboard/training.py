"""Training loop for the ErrorBoard FP8-add mouse experiment.

Implements task_spec.md §5 (sample weighting, three eval streams) and methodology.md
training rhythm (AdamW + cosine + warmup, log-spaced checkpoints, determinism).

Persistence: every run writes to runs/<run_name>/ with config.json, metrics.jsonl,
train.log (via tee in the launcher), checkpoint_<iter>.pt, STATUS, and PID files.
A future Claude/tmux session can re-orient from these files alone -- see status.py.
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

from .dataset import (
    EvalBatcher,
    StratifiedSampler,
    natural_distribution_weights,
    per_regime_eval_loaders,
)
from .model import GPT, GPTConfig
from . import preprocess as _add_preprocess
from . import preprocess_mult as _mul_preprocess
from . import regimes as _add_regimes
from . import mult_regimes as _mul_regimes
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer


def _resolve_tokenizer(name: str):
    """Return the tokenizer module matching `name`."""
    if name == "bit":
        return _bit_tokenizer
    if name == "sem":
        return _sem_tokenizer
    raise ValueError(f"unknown tokenization mode: {name!r}")


def _resolve_operation(name: str):
    """Return (preprocess_module, regimes_module) for the named operation."""
    if name == "add":
        return _add_preprocess, _add_regimes
    if name == "mul":
        return _mul_preprocess, _mul_regimes
    raise ValueError(f"unknown operation: {name!r}")


@dataclass
class TrainingConfig:
    """Per-run training configuration. Serialized to runs/<run_name>/config.json."""

    # run identification
    run_name: str = "default"
    runs_dir: str = "runs"
    seed: int = 0

    # model (forwarded to GPTConfig)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    d_mlp: int = 512
    pos_encoding: str = "learned"
    init_std: float = 0.02

    # tokenization: "bit" (default, 8-token-per-FP8) or "sem" (3-token-per-FP8).
    tokenization: str = "bit"

    # operation: "add" (default) or "mul"; selects which oracle and regime
    # classifier the pair table is built from.
    operation: str = "add"

    # optimizer (AdamW)
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
    train_eval_subsample: int = 256  # per-regime subsample size for training-loss stream
    checkpoint_iters: tuple = (100, 250, 500, 1000, 2500, 5000, 10000, 20000)

    # device / precision
    device: str = "cuda"

    # provenance (captured at run start)
    git_commit: str = field(default="")


# ---- helpers ----

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
    # use_deterministic_algorithms(True) breaks F.scaled_dot_product_attention's backward
    # on some configs; the seeded RNG plus cudnn.deterministic is sufficient for
    # our run-to-run reproducibility.


def _cosine_lr(iter_num: int, cfg: TrainingConfig) -> float:
    if iter_num < cfg.warmup_iters:
        return cfg.learning_rate * (iter_num + 1) / cfg.warmup_iters
    progress = (iter_num - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def _result_only_targets(targets: torch.Tensor, result_start: int, result_end: int) -> torch.Tensor:
    """Mask all non-result targets to -1 so cross_entropy / accuracy use result tokens only."""
    masked = targets.clone()
    n_target = targets.shape[1]
    keep = torch.zeros(n_target, dtype=torch.bool, device=targets.device)
    keep[result_start:result_end] = True
    masked[:, ~keep] = -1
    return masked


@torch.no_grad()
def _eval_pool(model: GPT, loader: EvalBatcher, device: torch.device,
               result_start: int, result_end: int) -> tuple[float, float, int]:
    """Iterate a loader and return (mean_result_loss, result_token_accuracy, n_samples).

    Loss/accuracy are computed over result tokens only.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    for batch in loader:
        inp = torch.from_numpy(batch["input"]).long().to(device)
        tgt = torch.from_numpy(batch["target"]).long().to(device)
        tgt_masked = _result_only_targets(tgt, result_start, result_end)
        logits, loss = model(inp, tgt_masked)
        valid = (tgt_masked != -1)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        total_loss += float(loss.item()) * n_valid
        total_tokens += n_valid
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == tgt) & valid).sum().item())
        n_samples += inp.shape[0]
    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan"), 0
    return total_loss / total_tokens, total_correct / total_tokens, n_samples


def _train_subsample_loaders(
    table: np.ndarray,
    train_indices: np.ndarray,
    batch_size: int,
    n_per_regime: int,
    num_regimes: int,
    encode_fn=None,
) -> dict[int, EvalBatcher]:
    """Per-regime training-pool eval loaders, deterministically subsampled to n_per_regime each."""
    out: dict[int, EvalBatcher] = {}
    for r in range(num_regimes):
        mask = table["regime_id"][train_indices] == r
        idx = train_indices[mask][:n_per_regime]
        if len(idx) > 0:
            out[r] = EvalBatcher(table, idx, batch_size, encode_fn=encode_fn)
    return out


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "STATUS").write_text(status + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---- main training loop ----

def train(cfg: TrainingConfig) -> dict:
    """Run a single training session per cfg. Writes everything to runs/<cfg.run_name>/."""
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    # Truncate metrics file on fresh run (re-running a run name overwrites).
    metrics_path.write_text("")

    cfg.git_commit = _git_commit()
    with open(run_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    (run_dir / "PID").write_text(f"{os.getpid()}\n")
    _write_status(run_dir, "running")

    try:
        _set_determinism(cfg.seed)
        device = torch.device(cfg.device)

        # ---- tokenization-spec resolution ----
        tk = _resolve_tokenizer(cfg.tokenization)
        seq_len = tk.SEQ_LEN
        vocab_size = tk.VOCAB_SIZE
        result_start = tk.POS_C_START - 1  # target-tensor offset
        result_end = tk.POS_C_END - 1
        encode_fn = tk.encode_batch

        # ---- operation dispatch (selects preprocess + regime modules) ----
        pp, rg = _resolve_operation(cfg.operation)
        regime_names = rg.REGIME_NAMES
        num_regimes = rg.NUM_REGIMES

        # ---- data ----
        table = pp.build_table()
        train_idx, holdout_idx = pp.split_train_holdout(
            table, holdout_frac=cfg.holdout_frac,
            min_holdout=cfg.min_holdout, seed=cfg.seed,
        )
        sampler = StratifiedSampler(table, train_idx, seed=cfg.seed, encode_fn=encode_fn)
        holdout_eval_loaders = per_regime_eval_loaders(
            table, holdout_idx, cfg.eval_batch_size, encode_fn=encode_fn,
        )
        train_eval_loaders = _train_subsample_loaders(
            table, train_idx, cfg.eval_batch_size, cfg.train_eval_subsample,
            num_regimes=num_regimes, encode_fn=encode_fn,
        )
        nat_weights = natural_distribution_weights(table)

        # ---- model ----
        gpt_config = GPTConfig(
            block_size=seq_len - 1,
            vocab_size=vocab_size,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, d_mlp=cfg.d_mlp,
            pos_encoding=cfg.pos_encoding, init_std=cfg.init_std,
        )
        model = GPT(gpt_config).to(device)
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

        # ---- training loop ----
        print(
            f"[{cfg.run_name}] model={n_params:,} params  pos_encoding={cfg.pos_encoding}  "
            f"tokenization={cfg.tokenization}  vocab={vocab_size}  seq_len={seq_len}  "
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
                    lo, ac, _ = _eval_pool(model, loader, device, result_start, result_end)
                    holdout_loss[regime_names[r]] = lo
                    holdout_acc[regime_names[r]] = ac
                row["holdout_loss"] = holdout_loss
                row["holdout_acc"] = holdout_acc

                # Natural-distribution scalar: weighted avg of per-regime holdout losses.
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
                    lo, _, _ = _eval_pool(model, loader, device, result_start, result_end)
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
                            "gpt_config": dataclasses.asdict(gpt_config),
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
            inp = torch.from_numpy(batch["input"]).long().to(device)
            tgt = torch.from_numpy(batch["target"]).long().to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inp, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        _write_status(run_dir, "completed")
        print(f"[{cfg.run_name}] completed in {time.time() - t_start:.0f}s", flush=True)
        return {"status": "completed", "final_iter": cfg.max_iters, "n_params": n_params}

    except KeyboardInterrupt:
        _write_status(run_dir, "killed")
        print(f"[{cfg.run_name}] killed by keyboard interrupt", flush=True)
        raise
    except Exception:
        _write_status(run_dir, "failed")
        traceback.print_exc()
        raise


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train an ErrorBoard mouse model.")
    # run identification
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    # model
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--d-mlp", type=int, default=512)
    p.add_argument("--pos-encoding", default="learned", choices=["learned", "rope"])
    p.add_argument("--tokenization", default="bit", choices=["bit", "sem"])
    p.add_argument("--operation", default="add", choices=["add", "mul"])
    # schedule
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    # eval
    p.add_argument("--eval-interval", type=int, default=250)
    # device
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = TrainingConfig(
        run_name=args.run_name, runs_dir=args.runs_dir, seed=args.seed,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, d_mlp=args.d_mlp,
        pos_encoding=args.pos_encoding, tokenization=args.tokenization,
        operation=args.operation,
        max_iters=args.max_iters, warmup_iters=args.warmup_iters,
        batch_size=args.batch_size, learning_rate=args.learning_rate, min_lr=args.min_lr,
        eval_interval=args.eval_interval, device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
