# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## End-to-end training

Train and save a TinyStories tokenizer:

```sh
uv run python -m cs336_basics.train_bpe \
  data/TinyStoriesV2-GPT4-train.txt \
  data/tinystories_tokenizer \
  --vocab-size 10000 \
  --special-token '<|endoftext|>'
```

Encode the train and validation splits as memory-mapped binary token arrays:

```sh
uv run python -m cs336_basics.prepare_data \
  data/TinyStoriesV2-GPT4-train.txt data/tinystories_train.bin \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt

uv run python -m cs336_basics.prepare_data \
  data/TinyStoriesV2-GPT4-valid.txt data/tinystories_valid.bin \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt
```

Start training on a CUDA server:

```sh
uv run python -m cs336_basics.train \
  --train-data data/tinystories_train.bin \
  --valid-data data/tinystories_valid.bin \
  --checkpoint checkpoints/tinystories.pt \
  --device cuda --amp-dtype bfloat16 --compile
```

Resume by adding `--resume checkpoints/tinystories.pt`. The model configuration
is written next to the checkpoint as a JSON file. Generate text after training:

```sh
uv run python -m cs336_basics.generate \
  --checkpoint checkpoints/tinystories.pt \
  --config checkpoints/tinystories.json \
  --vocab data/tinystories_tokenizer/vocab.json \
  --merges data/tinystories_tokenizer/merges.txt \
  --prompt 'Once upon a time'
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

