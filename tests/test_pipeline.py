import json
from fractions import Fraction

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM

from spinloop.cli import make_parser
from spinloop.data import parse_answer, prepare_gsm8k, prepare_hops, read_rows, write_rows

from test_model import tiny_config


def run(*argv):
    args = make_parser().parse_args([str(value) for value in argv])
    args.func(args)


def test_answers():
    assert parse_answer("text\n#### -1,200.50", "number") == Fraction("-1200.5")
    assert parse_answer("#### .25", "number") == Fraction(1, 4)
    assert parse_answer("#### 2/4", "number") == Fraction(1, 2)
    assert parse_answer("equals 3", "number") is None
    assert parse_answer("equals 3", "number", strict=False) == 3
    assert parse_answer("#### H", "label") == "H"


def test_data_splits(tmp_path):
    training = tmp_path / "source_train.jsonl"
    testing = tmp_path / "source_test.jsonl"
    write_rows(training, [{"question": f"Question {index}", "answer": "#### 1"} for index in range(7)])
    write_rows(testing, [{"question": "Question 0", "answer": "#### 1"}])
    output = tmp_path / "data"
    prepare_gsm8k(output, train_path=training, test_path=testing, dev_size=2, pilot_size=2)
    sets = {name: {row["id"] for row in read_rows(output / f"{name}.jsonl")}
            for name in ("train", "test", "dev", "pilot")}
    assert sets["pilot"] <= sets["train"]
    assert not (sets["train"] & sets["test"] or sets["train"] & sets["dev"] or sets["dev"] & sets["test"])
    prepare_hops(output, train_per_hop=2, eval_per_hop=2)
    split_graphs = []
    for split in ("train", "dev", "test"):
        graphs = set()
        for row in read_rows(output / f"{split}.jsonl"):
            mapping_text, rest = row["question"].split(". Start at ")
            mapping = dict(pair.split(" -> ") for pair in mapping_text.removeprefix("Mapping: ").split(", "))
            endpoint = rest[0]
            for _ in range(row["hops"]):
                endpoint = mapping[endpoint]
            assert endpoint == row["target"]
            graphs.add(row["id"].split("-")[1])
        assert not any(graphs & other for other in split_graphs)
        split_graphs.append(graphs)


def test_offline_cli(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = tmp_path / "base"
        Qwen3ForCausalLM(tiny_config()).save_pretrained(checkpoint)
        words = ["[PAD]", "[UNK]", "[EOS]", "####", "1", "2", "3", "add", "to"]
        raw = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, bos_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]",
        )
        tokenizer.chat_template = "{% for m in messages %}{{ m.content }} {% endfor %}"
        tokenizer.save_pretrained(checkpoint)
        data = tmp_path / "data.jsonl"
        write_rows(data, [
            {"id": "a", "question": "add 1 to 1", "target": "2", "solution": "#### 2", "task": "number"},
            {"id": "b", "question": "add 1 to 2", "target": "3", "solution": "#### 3", "task": "number"},
        ])
        adapter = tmp_path / "adapter"
        run("train", "--model", checkpoint, "--data", data, "--out", adapter, "--local-files-only", "--block", 8)
        base_result = tmp_path / "base-result"
        tune_result = tmp_path / "tune-result"
        run("eval", "--model", checkpoint, "--data", data, "--out", base_result, "--max-new", 3, "--local-files-only")
        run("eval", "--adapter", adapter, "--data", data, "--out", tune_result, "--depth", 2, "--max-new", 3,
            "--local-files-only", "--score-prefix")
        assert json.loads((tune_result / "eval_summary.json").read_text())["prefix_tokens"] > 0
        run("verify", "--adapter", adapter, "--cases", 2, "--local-files-only")
        run("bench", "--adapter", adapter, "--depth", 2, "--prompt-tokens", 3, "--new-tokens", 2,
            "--repeats", 1, "--local-files-only")
        comparison = tmp_path / "comparison.json"
        run("compare", "--baseline", base_result / "predictions.jsonl", "--variant",
            tune_result / "predictions.jsonl", "--out", comparison, "--bootstrap", 20)
        assert json.loads(comparison.read_text())["n"] == 2
    finally:
        torch.set_num_threads(old_threads)
