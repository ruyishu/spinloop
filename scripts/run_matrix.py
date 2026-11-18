"""Run the pilot or core matrix on one GPU, with explicitly selected evaluation splits."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command, directory, completion, reuse=False):
    if (directory / completion).is_file() and reuse:
        manifest = directory / "command.json"
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != command:
            raise RuntimeError(f"Existing run has a different configuration: {directory}")
        return
    subprocess.run([sys.executable, "-m", "spinloop.cli", *command], check=True)
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")


def run_seed(args):
    pilot = args.stage == "pilot"
    root = Path(args.out) / f"seed-{args.seed}"
    data = Path(args.data)
    options = ["--model", args.model, "--dtype", args.dtype]
    if args.revision:
        options += ["--revision", args.revision]
    if args.local_files_only:
        options += ["--local-files-only"]
    if args.device:
        options += ["--device", args.device]
    training = {
        "single-spin": ["--rotate-first", "--min-depth", "1", "--max-depth", "1", "--mode", "spin"],
        "loop-lora": ["--mode", "lora"],
        "spin": ["--mode", "spin"],
    }
    if not pilot:
        training["fixed"] = ["--mode", "fixed-spin"]
        training["no-feedback"] = ["--mode", "spin", "--alpha", "0"]
        for name in training.keys() - {"single-spin"}:
            training[name] += ["--max-depth", "4"]
    data_train = data / ("pilot.jsonl" if pilot else "train.jsonl")
    for name, treatment in training.items():
        directory = root / name
        command = [
            "train", *options, "--data", str(data_train), "--out", str(directory),
            "--seed", str(args.seed), "--lr", str(args.lr), "--epochs", str(args.epochs),
            "--microbatch", str(args.microbatch), "--checkpoint-passes", *treatment,
        ]
        if args.eval_only:
            if not (directory / "adapter.safetensors").is_file():
                raise FileNotFoundError(f"No trained adapter in {directory}")
        else:
            run(command, directory, "train_summary.json", reuse=True)

    matrix = [("original", 1, None, ()), ("single-spin", 1, "single-spin", ())]
    if pilot:
        matrix.append(("no-rotation", 2, None, ()))
    for name in training:
        if name == "single-spin":
            continue
        for depth in ([2] if pilot else [2, 3, 4]):
            matrix.append((name, depth, name, ()))
    if not pilot and args.extrapolate:
        for depth in (6, 8):
            matrix.extend((("spin", depth, "spin", ()),
                           ("clamped", depth, "spin", ("--clamp", "4")),
                           ("fixed", depth, "fixed", ())))
        matrix.extend((("constant-at-test", 4, "spin", ("--constant",)),
                       ("reversed-at-test", 4, "spin", ("--reverse",)),
                       ("zero-feedback-at-test", 4, "spin", ("--alpha", "0"))))

    eval_data = args.eval_data or str(data / f"{args.split}.jsonl")
    eval_root = root / args.split
    for name, depth, adapter, flags in matrix:
        directory = eval_root / f"{name}-d{depth}"
        command = [
            "eval", *options, "--data", eval_data, "--out", str(directory),
            "--depth", str(depth), "--max-new", str(args.max_new), "--seed", str(args.seed), *flags,
        ]
        if adapter:
            command += ["--adapter", str(root / adapter)]
        if args.limit:
            command += ["--limit", str(args.limit)]
        run(command, directory, "eval_summary.json", reuse=True)
    reference = ("original-d1", "no-rotation-d2", "single-spin-d1", "loop-lora-d2") if pilot else (
        "fixed-d4", "loop-lora-d4", "no-feedback-d4", "single-spin-d1",
    )
    main = "spin-d2" if pilot else "spin-d4"
    for left in reference:
        subprocess.run([
            sys.executable, "-m", "spinloop.cli", "compare",
            "--baseline", str(eval_root / left / "predictions.jsonl"),
            "--variant", str(eval_root / main / "predictions.jsonl"),
            "--out", str(eval_root / f"compare-{main}-vs-{left}.json"),
        ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "core"])
    parser.add_argument("--data", default="data/gsm8k")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--eval-data", help="Evaluate on this file; use a distinct --split for its output")
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--extrapolate", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.eval_only:
        parser.error("Create and select configurations before evaluating the official test split")
    if args.extrapolate and args.stage != "core":
        parser.error("Extrapolation requires the 2–4 pass training core matrix")
    run_seed(args)


if __name__ == "__main__":
    main()
