"""SpinLoop command-line entry points."""
import argparse

from .compare import compare
from .data import prepare_gsm8k, prepare_hops, prepare_svamp
from .experiment import evaluate, train
from .verify import benchmark, verify


def runtime_options(parser):
    parser.add_argument("--model", help="Model ID or local Qwen3 checkpoint")
    parser.add_argument("--revision", help="Pin a model commit")
    parser.add_argument("--device", help="Default: CUDA if available, otherwise CPU")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")


def eval_options(parser):
    runtime_options(parser)
    parser.add_argument("--adapter", help="Adapter run directory; omit for unmodified backbone")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--alpha", type=float, help="Override feedback strength for an ablation")
    parser.add_argument("--clamp", type=int, help="Reuse the rotation of this depth in later passes")
    parser.add_argument("--constant", action="store_true", help="Use R in all adapted passes")
    parser.add_argument("--reverse", action="store_true", help="Reverse the order of non-first rotations")


def make_parser():
    parser = argparse.ArgumentParser(prog="spinloop")
    sub = parser.add_subparsers(required=True)

    cmd = sub.add_parser("prepare-gsm8k")
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--train-jsonl")
    cmd.add_argument("--test-jsonl")
    cmd.add_argument("--revision")
    cmd.add_argument("--seed", type=int, default=2026)
    cmd.add_argument("--dev-size", type=int, default=512)
    cmd.add_argument("--pilot-size", type=int, default=2048)
    cmd.set_defaults(func=lambda a: prepare_gsm8k(a.out, train_path=a.train_jsonl, test_path=a.test_jsonl,
                                                  revision=a.revision, seed=a.seed, dev_size=a.dev_size,
                                                  pilot_size=a.pilot_size))
    cmd = sub.add_parser("prepare-svamp")
    cmd.add_argument("--input", required=True)
    cmd.add_argument("--out", required=True)
    cmd.set_defaults(func=lambda a: prepare_svamp(a.input, a.out))

    cmd = sub.add_parser("prepare-hops")
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--seed", type=int, default=2026)
    cmd.add_argument("--train-per-hop", type=int, default=1024)
    cmd.add_argument("--eval-per-hop", type=int, default=500)
    cmd.set_defaults(func=lambda a: prepare_hops(a.out, a.seed, a.train_per_hop, a.eval_per_hop))

    cmd = sub.add_parser("train")
    runtime_options(cmd)
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--mode", choices=["spin", "fixed-spin", "lora"], default="spin")
    cmd.add_argument("--block", type=int, default=16)
    cmd.add_argument("--rank-q", type=int, default=3)
    cmd.add_argument("--rank-v", type=int)
    cmd.add_argument("--alpha", type=float, default=1.0)
    cmd.add_argument("--rotate-first", action="store_true")
    cmd.add_argument("--min-depth", type=int, default=2)
    cmd.add_argument("--max-depth", type=int, default=2)
    cmd.add_argument("--checkpoint-passes", action="store_true")
    cmd.add_argument("--max-length", type=int, default=1024)
    cmd.add_argument("--batch-size", type=int, default=16)
    cmd.add_argument("--microbatch", type=int, default=1)
    cmd.add_argument("--epochs", type=int, default=1)
    cmd.add_argument("--lr", type=float, default=3e-4)
    cmd.add_argument("--warmup", type=float, default=0.05)
    cmd.add_argument("--weight-decay", type=float, default=0.0)
    cmd.add_argument("--seed", type=int, default=2026)
    cmd.set_defaults(func=train)

    cmd = sub.add_parser("eval")
    eval_options(cmd)
    cmd.add_argument("--data", required=True)
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--max-new", type=int, default=256)
    cmd.add_argument("--limit", type=int)
    cmd.add_argument("--seed", type=int, default=2026)
    cmd.add_argument("--samples", type=int, default=1)
    cmd.add_argument("--temperature", type=float, default=0.0)
    cmd.add_argument("--no-cache", action="store_true")
    cmd.add_argument("--score-prefix", action="store_true", help="Also score next tokens under the gold prefix")
    cmd.set_defaults(func=evaluate)

    cmd = sub.add_parser("compare")
    cmd.add_argument("--baseline", nargs="+", required=True)
    cmd.add_argument("--variant", nargs="+", required=True)
    cmd.add_argument("--bootstrap", type=int, default=5000)
    cmd.add_argument("--seed", type=int, default=2026)
    cmd.add_argument("--out")
    cmd.set_defaults(func=compare)

    cmd = sub.add_parser("verify")
    runtime_options(cmd)
    cmd.add_argument("--adapter")
    cmd.add_argument("--out")
    cmd.add_argument("--cases", type=int, default=32)
    cmd.set_defaults(func=verify)

    cmd = sub.add_parser("bench")
    eval_options(cmd)
    cmd.add_argument("--prompt-tokens", type=int, default=128)
    cmd.add_argument("--new-tokens", type=int, default=32)
    cmd.add_argument("--repeats", type=int, default=5)
    cmd.add_argument("--out")
    cmd.set_defaults(func=benchmark)
    return parser


def main():
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
