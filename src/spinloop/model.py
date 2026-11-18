"""Reuse the native Qwen3 stack, with a separate KV cache per loop."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Cayley(nn.Module):
    def __init__(self, width, block):
        super().__init__()
        if block < 2 or width % block:
            raise ValueError("block must be >= 2 and divide the input width")
        self.block = block
        self.blocks = width // block
        upper = torch.triu_indices(block, block, offset=1)
        self.register_buffer("rows", upper[0], persistent=False)
        self.register_buffer("cols", upper[1], persistent=False)
        self.entries = nn.Parameter(torch.zeros(self.blocks, upper.shape[-1]))

    def powers(self, limit):
        with torch.autocast(self.entries.device.type, enabled=False):
            upper = self.entries.float().new_zeros(self.blocks, self.block, self.block)
            upper[:, self.rows, self.cols] = self.entries.float()
            skew = upper - upper.transpose(-1, -2)
            identity = torch.eye(self.block, device=skew.device, dtype=skew.dtype)
            rotation = torch.linalg.solve(identity - skew, identity + skew)
            powers = {1: rotation}
            for exponent in range(2, limit + 1):
                powers[exponent] = powers[exponent - 1] @ rotation
        return powers

    def rotate(self, x, rotation):
        with torch.autocast(x.device.type, enabled=False):
            grouped = x.float().unflatten(-1, (self.blocks, self.block))
            rotated = torch.einsum("...bi,bji->...bj", grouped, rotation)
        return rotated.flatten(-2).to(x.dtype)


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.down = nn.Parameter(torch.empty(rank, in_features))
        self.up = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=5 ** 0.5)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.down), self.up)
        return delta.to(x.dtype)


@dataclass
class PassState:
    factor: int
    rotations: tuple
    residual: torch.Tensor | None = None


@dataclass
class LoopOutput:
    logits: torch.Tensor
    caches: tuple | None


class SpinLoop(nn.Module):
    def __init__(self, backbone, mode="spin", block=16, rank_q=3, rank_v=None,
                 alpha=1.0, rotate_first=False, checkpoint_passes=False):
        super().__init__()
        if backbone.config.model_type != "qwen3":
            raise ValueError("SpinLoop supports the dense text Qwen3 architecture")
        if mode not in {"none", "spin", "fixed-spin", "lora"}:
            raise ValueError(f"Unknown adapter mode: {mode}")
        if rotate_first and mode not in {"spin", "fixed-spin"}:
            raise ValueError("rotate_first requires a rotation adapter")
        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()
        self.mode = mode
        self.block = block
        self.alpha = alpha
        self.rank_q = rank_q
        self.rank_v = rank_v if rank_v is not None else (6 if backbone.config.hidden_size == 2048 else 3)
        self.rotate_first = rotate_first
        self.checkpoint_passes = checkpoint_passes
        self.adapters = nn.ModuleList()
        self._active = ContextVar("spinloop_pass", default=None)
        self._install_hooks()

    def _install_hooks(self):
        self.backbone.model.norm.register_forward_pre_hook(self._capture_residual)
        for layer in self.backbone.model.layers:
            for name, rank in (("q_proj", self.rank_q), ("v_proj", self.rank_v)):
                projection = getattr(layer.self_attn, name)
                index = len(self.adapters)
                if self.mode in {"spin", "fixed-spin"}:
                    self.adapters.append(Cayley(projection.in_features, self.block))
                    projection.register_forward_pre_hook(partial(self._rotate_input, index))
                elif self.mode == "lora":
                    self.adapters.append(LoRA(projection.in_features, projection.out_features, rank))
                    projection.register_forward_hook(partial(self._add_lora, index))

    def _capture_residual(self, module, inputs):
        state = self._active.get()
        if state is not None:
            state.residual = inputs[0]

    def _rotate_input(self, index, module, inputs):
        state = self._active.get()
        if state is not None and state.factor:
            return (self.adapters[index].rotate(inputs[0], state.rotations[index]),)
        return None

    def _add_lora(self, index, module, inputs, output):
        state = self._active.get()
        if state is not None and state.factor:
            return output + state.factor * self.adapters[index](inputs[0])
        return output

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def adapter_config(self):
        return {name: getattr(self, name) for name in
                ("mode", "block", "rank_q", "rank_v", "alpha", "rotate_first")}

    def depth_factors(self, depth, clamp=None, constant=False, reverse=False):
        if depth < 1 or (self.rotate_first and depth != 1):
            raise ValueError("depth must be positive, and single-pass SFT requires depth=1")
        if clamp is not None and clamp < 1:
            raise ValueError("clamp is a positive depth")
        factors = list(range(depth))
        if clamp is not None:
            factors = [min(factor, clamp - 1) for factor in factors]
        if self.mode == "fixed-spin" or constant:
            factors = [int(factor > 0) for factor in factors]
        if reverse:
            factors[1:] = reversed(factors[1:])
        if self.rotate_first:
            factors[0] = 1
        if self.mode == "none":
            factors = [0] * depth
        return tuple(factors)

    def prepare_rotations(self, factors):
        if not self.adapters or self.mode == "lora" or max(factors) == 0:
            return {}
        powers = [adapter.powers(max(factors)) for adapter in self.adapters]
        return {factor: tuple(table[factor] for table in powers)
                for factor in set(factors) if factor}

    def _stack(self, embeddings, attention_mask, position_ids, cache, use_cache,
               factor, rotations):
        state = PassState(factor, rotations)
        token = self._active.set(state)
        try:
            result = self.backbone.model(
                inputs_embeds=embeddings, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=cache, use_cache=use_cache,
            )
        finally:
            self._active.reset(token)
        if state.residual is None:
            raise RuntimeError("Cannot capture the Qwen3 residual before the final norm")
        return result.last_hidden_state, state.residual, result.past_key_values

    @staticmethod
    def _feedback(embeddings, residual, alpha):
        if alpha == 0:
            return embeddings
        source = embeddings.float()
        previous = residual.float()
        token_rms = source.square().mean(-1, keepdim=True).sqrt()
        residual_rms = previous.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        return (source + alpha * (token_rms / (residual_rms + 1e-6)) * previous).to(embeddings.dtype)

    def forward(self, input_ids, *, depth=2, attention_mask=None, position_ids=None,
                use_cache=False, caches=None, logit_mask=None, last_logits=False,
                factors=None, rotations=None, alpha=None):
        if caches is not None and (not use_cache or len(caches) != depth):
            raise ValueError("Reuse a cache only with the same depth and use_cache=True")
        factors = self.depth_factors(depth) if factors is None else factors
        if len(factors) != depth or not factors:
            raise ValueError("Need one adapter factor per pass")
        rotations = self.prepare_rotations(factors) if rotations is None else rotations
        embeddings = self.backbone.model.embed_tokens(input_ids)
        current = embeddings
        output_caches = []
        feedback_alpha = self.alpha if alpha is None else alpha
        for index, factor in enumerate(factors):
            layer_cache = caches[index] if caches is not None else None
            run_pass = partial(
                self._stack, attention_mask=attention_mask, position_ids=position_ids,
                cache=layer_cache, use_cache=use_cache, factor=factor,
                rotations=rotations.get(factor, ()),
            )
            if self.training and self.checkpoint_passes and not use_cache and factor:
                hidden, residual, new_cache = checkpoint(run_pass, current, use_reentrant=False)
            else:
                hidden, residual, new_cache = run_pass(current)
            if use_cache:
                output_caches.append(new_cache)
            if index + 1 < depth:
                current = self._feedback(embeddings, residual, feedback_alpha)
        if logit_mask is not None:
            hidden = hidden[logit_mask]
        elif last_logits:
            hidden = hidden[:, -1:, :]
        return LoopOutput(self.backbone.lm_head(hidden), tuple(output_caches) if use_cache else None)
