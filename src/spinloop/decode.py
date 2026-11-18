"""Fixed-depth decoding and synchronized timing on CPU or CUDA."""
import time

import torch


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, ids, *, depth, max_new, eos_ids, generator=None,
             temperature=0.0, cache=True, factors=None, alpha=None):
    if max_new < 1 or temperature < 0:
        raise ValueError("max_new must be positive and temperature nonnegative")
    device = ids.device
    sync(device)
    start = time.perf_counter()
    factors = model.depth_factors(depth) if factors is None else factors
    rotations = model.prepare_rotations(factors)
    tokens = []
    past = None
    current = ids
    prefill = 0.0
    for step in range(max_new):
        result = model(current, depth=depth, use_cache=cache, caches=past,
                       factors=factors, rotations=rotations, alpha=alpha, last_logits=True)
        logits = result.logits[0, -1].float()
        if temperature > 0:
            next_token = torch.multinomial((logits / temperature).softmax(-1), 1, generator=generator)
        else:
            next_token = logits.argmax().reshape(1)
        token = next_token.item()
        if step == 0:
            prefill = time.perf_counter() - start
        if token in eos_ids:
            break
        tokens.append(token)
        if cache:
            current = next_token.reshape(1, 1)
            past = result.caches
        else:
            current = torch.cat((current, next_token.reshape(1, 1)), dim=1)
    sync(device)
    return tokens, token not in eos_ids, step + 1, prefill, time.perf_counter() - start
