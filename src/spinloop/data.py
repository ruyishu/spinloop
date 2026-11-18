"""Dataset preparation, chat formatting, and deterministic answer scoring."""
import hashlib
import json
import random
import re
from fractions import Fraction
from pathlib import Path

import torch


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected nonempty rows with unique IDs in {path}")
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def identity(question):
    return hashlib.sha256(" ".join(question.casefold().split()).encode()).hexdigest()[:20]


def prepare_gsm8k(output, *, train_path=None, test_path=None, revision=None,
                  seed=2026, dev_size=512, pilot_size=2048):
    if bool(train_path) != bool(test_path):
        raise ValueError("Provide both original GSM8K JSONL files")
    if train_path:
        source = {}
        for split, path in (("train", train_path), ("test", test_path)):
            with Path(path).open(encoding="utf-8") as stream:
                source[split] = [json.loads(line) for line in stream if line.strip()]
    else:
        from datasets import load_dataset

        source = load_dataset("openai/gsm8k", "main", revision=revision)
    used = set()
    splits = {}
    for split in ("test", "train"):
        samples = []
        for row in source[split]:
            identifier = identity(row["question"])
            if identifier in used:
                continue
            used.add(identifier)
            if "####" not in row["answer"]:
                raise ValueError(f"GSM8K answer lacks a final marker: {identifier}")
            samples.append({
                "id": identifier, "question": row["question"],
                "solution": row["answer"], "target": row["answer"].rsplit("####", 1)[1].strip(),
                "task": "number",
            })
        splits[split] = samples
    train = splits["train"]
    if min(dev_size, pilot_size) < 1 or len(train) < dev_size + pilot_size:
        raise ValueError("Not enough distinct training questions for requested splits")
    random.Random(seed).shuffle(train)
    output = Path(output)
    splits["dev"] = train[:dev_size]
    splits["train"] = train[dev_size:]
    splits["pilot"] = splits["train"][:pilot_size]
    for name, samples in splits.items():
        write_rows(output / f"{name}.jsonl", samples)


def prepare_svamp(path, output):
    examples = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = []
    for row in examples:
        question = row["Body"].strip() + " " + row["Question"].strip()
        rows.append({"id": f"svamp-{row['ID']}", "question": question,
                     "target": str(row["Answer"]), "task": "number"})
    write_rows(output, rows)


def prepare_hops(output, seed=2026, train_per_hop=1024, eval_per_hop=500):
    if min(train_per_hop, eval_per_hop) < 1:
        raise ValueError("Sample counts must be positive")
    rng = random.Random(seed)
    labels = list("ABCDEFGHIJKLMNOP")
    seen = set()

    def examples(hops, count, split):
        for hop in hops:
            targets = [labels[index % len(labels)] for index in range(count)]
            rng.shuffle(targets)
            for target in targets:
                while True:
                    ring = labels[:1] + rng.sample(labels[1:], len(labels) - 1)
                    signature = "".join(ring)
                    if signature not in seen:
                        seen.add(signature)
                        break
                mapping = list(zip(ring, ring[1:] + ring[:1]))
                rng.shuffle(mapping)
                start = ring[(ring.index(target) - hop) % len(ring)]
                graph = ", ".join(f"{key} -> {value}" for key, value in mapping)
                question = f"Mapping: {graph}. Start at {start} and follow exactly {hop} edges. Where do you end?"
                yield {"id": f"{split}-{signature}", "question": question, "task": "label",
                       "target": target, "solution": f"#### {target}", "hops": hop}

    output = Path(output)
    write_rows(output / "train.jsonl", examples(range(1, 5), train_per_hop, "train"))
    write_rows(output / "dev.jsonl", examples(range(1, 5), eval_per_hop, "dev"))
    write_rows(output / "test.jsonl", examples(range(1, 9), eval_per_hop, "test"))


def prompt_ids(tokenizer, row):
    instruction = ("Output only the final line in the format #### LETTER."
                   if row["task"] == "label"
                   else "Show concise work, then put the answer on a separate last line as #### NUMBER.")
    messages = [{"role": "user", "content": row["question"] + "\n" + instruction}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def training_examples(tokenizer, rows, max_length):
    items = []
    for row in rows:
        prompt = prompt_ids(tokenizer, row)
        answer = tokenizer.encode(row["solution"].strip(), add_special_tokens=False)
        answer.append(tokenizer.eos_token_id)
        sequence = prompt + answer
        if len(sequence) <= max_length:
            items.append((sequence, [-100] * len(prompt) + answer, row["id"]))
    if not items:
        raise ValueError("No training examples fit the requested maximum length")
    return items


def collate(items, pad_id, device):
    width = max(len(example[0]) for example in items)
    ids, labels, mask = [], [], []
    for sequence, target, _ in items:
        pad = width - len(sequence)
        ids.append(sequence + [pad_id] * pad)
        labels.append(target + [-100] * pad)
        mask.append([1] * len(sequence) + [0] * pad)
    return tuple(torch.tensor(value, device=device, dtype=torch.long)
                 for value in (ids, labels, mask))


_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?|[+-]?\.\d+")


def parse_answer(text, task, strict=True):
    if strict:
        if "####" not in text:
            return None
        candidate = text.rsplit("####", 1)[1].strip().splitlines()
        candidate = candidate[0].strip() if candidate else ""
    else:
        candidate = text.strip()
    if task == "label":
        if strict:
            return candidate.upper() if re.fullmatch("[A-Pa-p]", candidate) else None
        matches = re.findall(r"\b[A-P]\b", candidate)
        return matches[-1] if matches else None
    candidate = candidate.replace("−", "-")
    match = _NUMBER.fullmatch(candidate) if strict else None
    if not strict:
        matches = list(_NUMBER.finditer(candidate))
        match = matches[-1] if matches else None
    try:
        return Fraction(match[0].replace(",", "")) if match else None
    except (ValueError, ZeroDivisionError):
        return None
