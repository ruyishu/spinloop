"""Model parity, causality, and fixed-shape timing checks."""
import json
import statistics

import torch

from .decode import generate
from .experiment import environment, runtime, write_json


@torch.inference_mode()
def check_loop(model, ids, depths=(1, 2, 4)):
    base = model.backbone(ids, use_cache=False).logits
    first = model(ids, depth=1).logits
    torch.testing.assert_close(first, base, rtol=0, atol=0)
    result = {"first_pass_max_error": float((base - first).abs().max()), "depths": {}}
    prefix = max(1, ids.shape[1] // 2)
    changed = ids.clone()
    changed[:, prefix:] = (changed[:, prefix:] + 1) % model.backbone.config.vocab_size
    bf16 = base.dtype == torch.bfloat16
    tolerance = {"atol": 0.1, "rtol": 0.03} if bf16 else {"atol": 3e-5, "rtol": 3e-5}
    for depth in depths:
        full = model(ids, depth=depth).logits
        intervened = model(changed, depth=depth).logits
        torch.testing.assert_close(full[:, :prefix], intervened[:, :prefix], **tolerance)
        chunks = []
        caches = None
        for index in range(ids.shape[1]):
            step = model(ids[:, index:index+1], depth=depth, use_cache=True, caches=caches)
            caches = step.caches
            if len({id(cache) for cache in caches}) != depth:
                raise AssertionError("Loop passes cannot share a KV cache")
            chunks.append(step.logits)
        incremental = torch.cat(chunks, dim=1)
        torch.testing.assert_close(full, incremental, **tolerance)
        result["depths"][depth] = {
            "cache_max_error": float((full - incremental).abs().max()),
            "cache_top1_matches": int((full.argmax(-1) == incremental.argmax(-1)).sum()),
            "positions": ids.numel(),
            "causal_max_error": float((full[:, :prefix] - intervened[:, :prefix]).abs().max()),
        }
    return result


def verify(args):
    if args.cases < 1:
        raise ValueError("cases must be positive")
    model, _, device, _, _ = runtime(args, checkpoint_path=args.adapter)
    model.eval()
    if model.rotate_first:
        raise ValueError("First-pass parity applies only to adapters gated off on the first pass")
    rng = torch.Generator(device=device).manual_seed(2026)
    report = {"runtime": environment(device, args), "cases": []}
    for index in range(args.cases):
        size = 2 + index % 16
        ids = torch.randint(model.backbone.config.vocab_size, (1, size), device=device, generator=rng)
        report["cases"].append(check_loop(model, ids))
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report), flush=True)


def benchmark(args):
    if min(args.prompt_tokens, args.new_tokens, args.repeats) < 1:
        raise ValueError("benchmark dimensions must be positive")
    model, _, device, _, _ = runtime(args, adapter_options={"mode": "none"}, checkpoint_path=args.adapter)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(2026)
    ids = torch.randint(model.backbone.config.vocab_size, (1, args.prompt_tokens),
                        device=device, generator=generator)
    factors = model.depth_factors(args.depth, clamp=args.clamp, constant=args.constant, reverse=args.reverse)
    options = {"depth": args.depth, "max_new": args.new_tokens, "eos_ids": set(),
               "alpha": args.alpha, "factors": factors}
    generate(model, ids, **options)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results = [generate(model, ids, **options) for _ in range(args.repeats)]
    report = {"runtime": environment(device, args), "depth_factors": factors, "batch_size": 1,
              "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens,
              "median_prefill_seconds": statistics.median(run[3] for run in results),
              "median_decode_seconds": statistics.median(run[4] - run[3] for run in results),
              "median_total_seconds": statistics.median(run[4] for run in results),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report), flush=True)
