"""Train adapters and evaluate fixed-depth or sampled decoding."""
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .data import collate, parse_answer, prompt_ids, read_rows, training_examples, write_rows
from .decode import generate, sync
from .model import SpinLoop

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime(args, adapter_options=None, checkpoint_path=None):
    metadata = None
    if checkpoint_path:
        metadata = json.loads((Path(checkpoint_path) / "adapter.json").read_text(encoding="utf-8"))
    model_id = args.model or (metadata["model"] if metadata else DEFAULT_MODEL)
    revision = args.revision or (metadata["revision"] if metadata else None)
    if metadata and (model_id != metadata["model"] or revision != metadata["revision"]):
        raise ValueError("Adapter must use the model and revision it was trained on")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    base = Qwen3ForCausalLM.from_pretrained(
        model_id, revision=revision, local_files_only=args.local_files_only,
        dtype=dtype, attn_implementation=args.attention,
    ).to(device)
    pinned_revision = getattr(base.config, "_commit_hash", None) or revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=pinned_revision, local_files_only=args.local_files_only,
    )
    model = SpinLoop(base, **(metadata["adapter"] if metadata else adapter_options or {})).to(device)
    if metadata:
        model.adapters.load_state_dict(load_file(str(Path(checkpoint_path) / "adapter.safetensors")))
    return model, tokenizer, device, model_id, pinned_revision


def environment(device, args):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dtype": args.dtype, "attention": args.attention}


def arguments(args):
    return {key: value for key, value in vars(args).items() if key != "func"}


def supervised_logits(model, batch, depth, *, factors=None, alpha=None):
    ids, labels, attention = batch
    targets = labels[:, 1:]
    valid_targets = targets.ne(-100)
    final = torch.zeros_like(valid_targets[:, :1])
    logit_mask = torch.cat((valid_targets, final), dim=1)
    logits = model(ids, depth=depth, attention_mask=attention, logit_mask=logit_mask,
                   factors=factors, alpha=alpha).logits
    return logits.float(), targets[valid_targets]


def loss_sum(model, batch, depth):
    logits, targets = supervised_logits(model, batch, depth)
    return F.cross_entropy(logits, targets, reduction="sum")


def train(args):
    if min(args.epochs, args.batch_size, args.microbatch, args.max_length) < 1:
        raise ValueError("Sizes and epochs must be positive")
    if args.mode == "none" or min(args.min_depth, args.max_depth) < 1 or args.min_depth > args.max_depth:
        raise ValueError("Training requires an adapter and a valid depth interval")
    if (args.min_depth == 1) != args.rotate_first or (args.rotate_first and args.max_depth != 1):
        raise ValueError("Train either only depth 1 with --rotate-first or depths >= 2")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new run directory: {out}")
    torch.manual_seed(args.seed)
    model, tokenizer, device, model_id, revision = runtime(args, {
        "mode": args.mode, "block": args.block, "rank_q": args.rank_q,
        "rank_v": args.rank_v, "alpha": args.alpha, "rotate_first": args.rotate_first,
        "checkpoint_passes": args.checkpoint_passes,
    })
    model.train()
    rows = read_rows(args.data)
    examples = training_examples(tokenizer, rows, args.max_length)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    params = list(model.adapters.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    depth_rng = random.Random(args.seed + 1)
    updates = math.ceil(len(examples) / args.batch_size) * args.epochs
    warmup = math.ceil(updates * args.warmup)
    processed_tokens = 0
    supervised_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    logs = []
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for start in range(0, len(examples), args.batch_size):
            window = examples[start:start + args.batch_size]
            depth = depth_rng.randint(args.min_depth, args.max_depth)
            total_targets = sum(sum(label != -100 for label in item[1][1:]) for item in window)
            supervised_tokens += total_targets
            processed_tokens += sum(len(item[0]) for item in window) * depth
            step = len(logs) + 1
            lr_scale = min(1.0, step / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for offset in range(0, len(window), args.microbatch):
                batch = collate(window[offset:offset + args.microbatch], pad_id, device)
                total_loss = loss_sum(model, batch, depth)
                (total_loss / total_targets).backward()
                value += total_loss.detach().item()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
            optimizer.step()
            logs.append({"step": step, "epoch": epoch + 1, "depth": depth,
                         "loss": value / total_targets, "gradient_norm": norm, "lr": args.lr * lr_scale})
            if step == 1 or step == updates or step % 25 == 0:
                print(json.dumps(logs[-1]), flush=True)
    sync(device)
    train_seconds = time.perf_counter() - started
    out.mkdir(parents=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.adapters.state_dict().items()}
    save_file(state, str(out / "adapter.safetensors"))
    write_json(out / "adapter.json", {"adapter": model.adapter_config(), "model": model_id, "revision": revision})
    write_rows(out / "training.jsonl", logs)
    write_json(out / "train_summary.json", {
        "args": arguments(args), "runtime": environment(device, args), "model": model_id, "revision": revision,
        "dataset_sha256": sha256(args.data), "train_seconds": train_seconds,
        "supervised_tokens": supervised_tokens, "processed_depth_tokens": processed_tokens,
        "trainable_parameters": sum(p.numel() for p in params),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "included_ids": sorted(item[2] for item in examples), "input_count": len(rows),
    })


def evaluation_question(model, tokenizer, row, args, device, factors, rng):
    tokens = torch.tensor([prompt_ids(tokenizer, row)], device=device)
    eos = model.backbone.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tokenizer.eos_token_id)
    attempts = []
    for _ in range(args.samples):
        generated, truncated, steps, prefill, seconds = generate(
            model, tokens, depth=args.depth, max_new=args.max_new, eos_ids=eos, generator=rng,
            temperature=args.temperature, cache=not args.no_cache, factors=factors, alpha=args.alpha,
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        attempts.append({"text": text, "strict": parse_answer(text, row["task"]),
                         "loose": parse_answer(text, row["task"], strict=False), "truncated": truncated,
                         "tokens": len(generated), "steps": steps, "prefill_seconds": prefill, "seconds": seconds})
    votes = Counter(attempt["strict"] for attempt in attempts if attempt["strict"] is not None)
    winner = votes.most_common(1)[0][0] if votes else None
    selected = next(attempt for attempt in attempts if attempt["strict"] == winner)
    target = parse_answer("#### " + str(row["target"]), row["task"])
    if target is None:
        raise ValueError(f"Unparseable gold answer for {row['id']}: {row['target']!r}")
    prediction = {
        "id": row["id"], "task": row["task"], "hops": row.get("hops"),
        "target": str(target), "answer": str(winner) if winner is not None else None,
        "correct": winner == target, "loose_correct": selected["loose"] == target,
        "formatted": winner is not None, "truncated": selected["truncated"], "prompt_tokens": tokens.numel(),
        "completion_tokens": sum(attempt["tokens"] for attempt in attempts),
        "decode_steps": sum(attempt["steps"] - 1 for attempt in attempts),
        "prefill_seconds": sum(attempt["prefill_seconds"] for attempt in attempts),
        "seconds": sum(attempt["seconds"] for attempt in attempts),
    }
    prediction["attempts"] = [
        {key: (str(value) if value is not None else None) if key in {"strict", "loose"} else value
         for key, value in attempt.items()} for attempt in attempts
    ]
    if args.score_prefix and "solution" in row:
        example = training_examples(tokenizer, [row], model.backbone.config.max_position_embeddings)
        with torch.inference_mode():
            batch = collate(example, tokenizer.eos_token_id, device)
            logits, targets = supervised_logits(model, batch, args.depth, factors=factors, alpha=args.alpha)
            prediction["prefix_nll_sum"] = F.cross_entropy(logits, targets, reduction="sum").item()
            prediction["prefix_top1_count"] = (logits.argmax(-1) == targets).sum().item()
            prediction["prefix_tokens"] = targets.numel()
    return prediction


def evaluate(args):
    if args.samples < 1 or (args.samples > 1 and args.temperature <= 0):
        raise ValueError("Multiple samples require a positive temperature")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new run directory: {out}")
    model, tokenizer, device, model_id, revision = runtime(
        args, adapter_options={"mode": "none"}, checkpoint_path=args.adapter,
    )
    model.eval()
    rows = read_rows(args.data)[:args.limit]
    factors = model.depth_factors(args.depth, clamp=args.clamp,
                                  constant=args.constant, reverse=args.reverse)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    predictions = []
    for index, row in enumerate(rows, start=1):
        predictions.append(evaluation_question(model, tokenizer, row, args, device, factors, rng))
        if index % 25 == 0:
            print(f"Evaluated {index}/{len(rows)}", flush=True)
    out.mkdir(parents=True)
    write_rows(out / "predictions.jsonl", predictions)
    n = len(predictions)
    seconds = sum(row["seconds"] for row in predictions)
    prefill = sum(row["prefill_seconds"] for row in predictions)
    completion = sum(row["completion_tokens"] for row in predictions)
    groups = sorted({row["hops"] for row in predictions if row["hops"] is not None})
    summary = {
        "args": arguments(args), "runtime": environment(device, args), "n": n,
        "model": model_id, "revision": revision, "dataset_sha256": sha256(args.data),
        "adapter_sha256": sha256(Path(args.adapter) / "adapter.safetensors") if args.adapter else None,
        "depth_factors": factors, "accuracy": sum(row["correct"] for row in predictions) / n,
        "loose_accuracy": sum(row["loose_correct"] for row in predictions) / n,
        "format_rate": sum(row["formatted"] for row in predictions) / n,
        "truncation_rate": sum(row["truncated"] for row in predictions) / n,
        "mean_completion_tokens_per_attempt": completion / (n * args.samples),
        "total_seconds": seconds, "prefill_seconds": prefill, "decode_seconds": seconds - prefill,
        "completion_tokens_per_second": completion / seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "accuracy_by_hop": {
            hop: sum(row["correct"] for row in predictions if row["hops"] == hop)
            / sum(row["hops"] == hop for row in predictions) for hop in groups
        },
    }
    scored_tokens = sum(row.get("prefix_tokens", 0) for row in predictions)
    if scored_tokens:
        summary["prefix_tokens"] = scored_tokens
        summary["prefix_nll"] = sum(row.get("prefix_nll_sum", 0) for row in predictions) / scored_tokens
        summary["prefix_top1"] = sum(row.get("prefix_top1_count", 0) for row in predictions) / scored_tokens
    write_json(out / "eval_summary.json", summary)
    print(json.dumps(summary), flush=True)
