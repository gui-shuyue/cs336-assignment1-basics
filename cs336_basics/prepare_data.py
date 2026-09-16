from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import BPETokenizer


def encode_file(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    dtype: str | np.dtype,
    buffer_size: int = 1_000_000,
) -> int:
    """Stream a UTF-8 text file into a headerless binary token array."""
    dtype = np.dtype(dtype)
    max_token_id = max(tokenizer.vocab)
    if max_token_id > np.iinfo(dtype).max:
        raise ValueError(f"dtype {dtype} cannot represent maximum token ID {max_token_id}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_buffer: list[int] = []
    total_tokens = 0

    with open(input_path, encoding="utf-8") as source, open(output_path, "wb") as target:
        for line in source:
            token_buffer.extend(tokenizer.encode(line))
            if len(token_buffer) >= buffer_size:
                np.asarray(token_buffer, dtype=dtype).tofile(target)
                total_tokens += len(token_buffer)
                token_buffer.clear()

        if token_buffer:
            np.asarray(token_buffer, dtype=dtype).tofile(target)
            total_tokens += len(token_buffer)

    metadata = {
        "input_path": os.fspath(input_path),
        "output_path": os.fspath(output_path),
        "dtype": dtype.name,
        "num_tokens": total_tokens,
        "vocab_size": len(tokenizer.vocab),
    }
    with open(f"{output_path}.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text into a binary token array.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(
        args.vocab,
        args.merges,
        args.special_tokens or ["<|endoftext|>"],
    )
    total = encode_file(tokenizer, args.input_path, args.output_path, args.dtype)
    print(f"Wrote {total:,} tokens to {args.output_path}")


if __name__ == "__main__":
    main()
