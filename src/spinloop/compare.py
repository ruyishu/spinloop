"""Paired changes across questions, optionally averaged across training seeds."""
import json
import random
import statistics

from .data import read_rows


def compare(args):
    if len(args.baseline) != len(args.variant) or args.bootstrap < 1:
        raise ValueError("Provide equally many baseline and variant files and positive bootstrap draws")
    ids = None
    targets = None
    differences = []
    details = []
    for baseline, variant in zip(args.baseline, args.variant):
        baseline_rows = {row["id"]: row for row in read_rows(baseline)}
        variant_rows = {row["id"]: row for row in read_rows(variant)}
        if baseline_rows.keys() != variant_rows.keys() or (ids is not None and set(ids) != set(baseline_rows)):
            raise ValueError("All paired prediction files must use the same question IDs")
        ids = sorted(baseline_rows)
        pair_targets = [(baseline_rows[key]["target"], baseline_rows[key]["task"]) for key in ids]
        other_targets = [(variant_rows[key]["target"], variant_rows[key]["task"]) for key in ids]
        if pair_targets != other_targets or (targets is not None and pair_targets != targets):
            raise ValueError("All pairs must have matching task and answer labels")
        targets = pair_targets
        base_correct = [bool(baseline_rows[key]["correct"]) for key in ids]
        variant_correct = [bool(variant_rows[key]["correct"]) for key in ids]
        delta = [target - original for original, target in zip(base_correct, variant_correct)]
        differences.append(delta)
        details.append({
            "baseline": baseline, "variant": variant, "baseline_accuracy": statistics.mean(base_correct),
            "variant_accuracy": statistics.mean(variant_correct), "delta_pp": 100 * statistics.mean(delta),
            "wrong_to_right": delta.count(1), "right_to_wrong": delta.count(-1),
        })
    per_question = [statistics.mean(values) for values in zip(*differences)]
    rng = random.Random(args.seed)
    draws = sorted(100 * sum(rng.choices(per_question, k=len(ids))) / len(ids)
                   for _ in range(args.bootstrap))
    report = {"n": len(ids), "pairs": details, "mean_delta_pp": 100 * statistics.mean(per_question),
              "seed_stdev_pp": statistics.stdev([row["delta_pp"] for row in details]) if len(details) > 1 else None,
              "question_bootstrap_95_pp": [
                  draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))],
              ]}
    if args.out:
        from .experiment import write_json

        write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)
