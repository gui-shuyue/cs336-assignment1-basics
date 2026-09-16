from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch, load_token_array
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


def _autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _compute_loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_batches: int,
    amp_dtype: str,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        with _autocast_context(device, amp_dtype):
            loss = _compute_loss(model, x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer LM.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data")
    parser.add_argument("--data-dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_data = load_token_array(args.train_data, args.data_dtype)
    valid_data = load_token_array(args.valid_data, args.data_dtype) if args.valid_data else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        layer_num=args.num_layers,
        head_num=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(args.resume, model, optimizer)

    raw_model = model
    if args.compile:
        model = torch.compile(model)

    use_scaler = device.type == "cuda" and args.amp_dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_path.with_suffix(".json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2)

    model.train()
    last_log_time = time.perf_counter()
    for iteration in range(start_iteration, args.steps):
        learning_rate = get_lr_cosine_schedule(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_steps,
            args.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        x, y = get_batch(
            train_data,
            args.batch_size,
            args.context_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, args.amp_dtype):
            loss = _compute_loss(model, x, y)

        scaler.scale(loss).backward()
        if use_scaler:
            scaler.unscale_(optimizer)
        gradient_clipping(raw_model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        completed = iteration + 1
        if completed % args.log_interval == 0 or completed == 1:
            now = time.perf_counter()
            elapsed = now - last_log_time
            tokens = args.batch_size * args.context_length * (1 if completed == 1 else args.log_interval)
            print(f"step={completed} loss={loss.item():.4f} lr={learning_rate:.3e} tokens/s={tokens / elapsed:,.0f}")
            last_log_time = now

        if valid_data is not None and args.eval_interval > 0 and completed % args.eval_interval == 0:
            validation_loss = estimate_loss(
                model,
                valid_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_batches,
                args.amp_dtype,
            )
            print(f"step={completed} validation_loss={validation_loss:.4f}")

        if args.checkpoint_interval > 0 and completed % args.checkpoint_interval == 0:
            save_checkpoint(raw_model, optimizer, completed, checkpoint_path)

    save_checkpoint(raw_model, optimizer, args.steps, checkpoint_path)
    print(f"Saved final checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
