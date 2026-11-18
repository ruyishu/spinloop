# SpinLoop

SpinLoop retrofits a frozen dense Qwen3 checkpoint with recurrent depth. It preserves the original computation at depth one and adapts Q and V input projections at later depths with a shared block-diagonal Cayley rotation. A four-pass run applies the sequence I, R, R², R³. The recurrent input carries forward the previous pass's residual plus the original token embedding. Training uses supervised next-token prediction on labeled answers.

The code uses the native Qwen3 transformer layers, keeps independent logical KV caches for each pass, and stores adapter-only checkpoints. The evaluation scripts test the rotation schedule, the contribution of hidden-state feedback, and untrained depths. See [the validation plan](docs/validation-plan.md) for the research design.

## Install

Use Python 3.10+ and install a CUDA-enabled PyTorch build for the server driver. The packaged experiment uses one GPU per process. On two GPUs, run independent jobs on different devices with `CUDA_VISIBLE_DEVICES`.

```sh
python -m pip install -e '.[datasets,test]'
python -m pytest -q
spinloop --help
```

The offline tests instantiate a tiny Qwen3 and run training, checkpoint reload, fixed-depth generation, and paired scoring. They test the pipeline without downloading a model. They do not measure trained Qwen3-0.6B or 1.7B accuracy.

## First GPU run

```sh
spinloop prepare-gsm8k --out data/gsm8k
spinloop verify --dtype bf16 --cases 32
python scripts/run_matrix.py pilot --out runs/pilot --limit 16
```

This trains on 2,048 official GSM8K training examples and evaluates 16 development questions to check all five groups. The 512-question development set is separate from training. To score the entire development set with the already trained adapters, use a fresh evaluation label.

```sh
python scripts/run_matrix.py pilot --out runs/pilot --split full-dev \
  --eval-data data/gsm8k/dev.jsonl --eval-only
spinloop verify --adapter runs/pilot/seed-2026/spin --dtype bf16 --cases 32
```

An existing output is reused only when its saved command matches. Choose separate output roots for different training settings. Use `--train-jsonl` and `--test-jsonl` on `prepare-gsm8k` for local copies of the original dataset. Use an immutable Hugging Face model commit with `--revision` and a dataset commit with the preparation command's `--revision` for reruns. Set `--local-files-only` for machines with predownloaded artifacts.

| Pilot run | Parameters | Depth |
|---|---:|---:|
| Original frozen Qwen3 | 0 | 1 |
| Frozen Qwen3 with recurrent input | 0 | 2 |
| Single-pass Spin SFT | 430,080 | 1 |
| Matched Q/V LoRA in the second pass | 430,080 | 2 |
| SpinLoop in the second pass | 430,080 | 2 |

The default rotation block is 16. The LoRA control uses Q/V ranks 3/3 on 0.6B and 3/6 on 1.7B, with 860,160 trainable parameters on the latter. Default training is 1 epoch with batch 16, microbatch 1, and learning rate 3e-4. For the second tuning opportunity, run the pilot into another directory with `--lr 1e-3` and select on development predictions. Check the per-question counts of wrong-to-right and right-to-wrong transitions before increasing depth.

## Confirmatory matrix

```sh
# Tune and select on the development split first.
python scripts/run_matrix.py core --out runs/core --seed 2026
# After fixing all settings, repeat for the other training seeds.
python scripts/run_matrix.py core --out runs/core --seed 2027
python scripts/run_matrix.py core --out runs/core --seed 2028
# Open the held-out test only after selection.
for seed in 2026 2027 2028; do
  python scripts/run_matrix.py core --out runs/core --seed "$seed" \
    --split test --eval-only --extrapolate
done
```

The core matrix includes constant rotation, power-scheduled rotation, shared depth-scaled LoRA, single-pass SFT, independently trained zero-feedback Spin, and the original backbone. The main comparison is SpinLoop versus constant rotation at depth 4. Trainable multi-pass groups use identical sample order and seeded depth draws from 2–4 within each seed. Extrapolation scores depths 6 and 8 with continued rotation, rotation clamped at depth 4, and the trained constant schedule. It includes inference-time feedback removal, constant rotation, and reversed schedule at depth 4.

```sh
spinloop compare \
  --baseline runs/core/seed-{2026,2027,2028}/test/fixed-d4/predictions.jsonl \
  --variant runs/core/seed-{2026,2027,2028}/test/spin-d4/predictions.jsonl \
  --out runs/core/primary.json
```

Paired bootstrap resamples question IDs together across seeds. The JSON includes each seed's raw transitions and the spread across seeds. The interval captures question-sampling uncertainty for those runs.

For scale replication use the same core commands with `--model Qwen/Qwen3-1.7B` and a fresh output root. For independent transfer, download [SVAMP.json](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json), run `spinloop prepare-svamp --input SVAMP.json --out data/svamp.jsonl`, and run the trained matrix with `--eval-only --split svamp --eval-data data/svamp.jsonl`. For controlled depth and task difficulty, run `spinloop prepare-hops --out data/hops`, train a separate core matrix with `--data data/hops --max-new 12`, then score its test split. The held-out graph generator includes hops 5–8. Each summary reports accuracy by hop.

## Measure latency and compute

```sh
spinloop bench --adapter runs/core/seed-2026/spin --depth 4 --dtype bf16 \
  --prompt-tokens 256 --new-tokens 64 --repeats 5 --out runs/core/bench-spin-d4.json
spinloop eval --adapter runs/core/seed-2026/single-spin --depth 1 --dtype bf16 \
  --data data/gsm8k/dev.jsonl --samples 4 --temperature 0.7 --out runs/budget-single-dev
```

The second command provides a starting point for compute matching, not a preselected budget. Calibrate sample count on the development set using measured prefill and decode time on the same GPU. Voting uses parsed answers and breaks ties by first occurrence. Set the sample count and temperature before test evaluation. The benchmark uses fixed input length, generated length, batch size 1, and a warmup. Per-question output records sampled answers, timing, length, format failure, and truncation. Run summaries include dataset and adapter hashes, runtime versions, GPU and peak memory, with supervised and processed tokens logged for training.

Use `spinloop eval --score-prefix` to separately record NLL and next-token correctness on identical gold answer prefixes. This requires examples with solutions, and the prefix-scoring pass is outside the generation timings.

The TaH/TaH2 public checkpoints can be evaluated separately with their [official runner](https://github.com/thu-nics/TaH). Their training data, iteration policy, and inference stack differ. The matched in-repo LoRA and constant-rotation groups support the controlled comparisons.
