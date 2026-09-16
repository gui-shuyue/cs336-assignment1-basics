from __future__ import annotations

import argparse
import json

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    stop_token: str | None = "<|endoftext|>",
) -> str:
    """Generate text autoregressively with optional nucleus sampling."""
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        if stop_token is None:
            raise ValueError("An empty prompt requires a stop token")
        token_ids = [tokenizer.byte_to_id[stop_token.encode("utf-8")]]

    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    stop_id = tokenizer.byte_to_id.get(stop_token.encode("utf-8")) if stop_token is not None else None

    model.eval()
    for _ in range(max_new_tokens):
        context = tokens[:, -model.context_length :]
        logits = model(context)[:, -1, :]

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = probabilities.sort(dim=-1, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                remove = cumulative - sorted_probs >= top_p
                sorted_probs = sorted_probs.masked_fill(remove, 0)
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1)

        tokens = torch.cat((tokens, next_token), dim=-1)
        if stop_id is not None and next_token.item() == stop_id:
            break

    return tokenizer.decode(tokens[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--special-token", action="append", dest="special_tokens")
    args = parser.parse_args()

    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    with open(args.config, encoding="utf-8") as config_file:
        config = json.load(config_file)

    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        layer_num=config["num_layers"],
        head_num=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config["rope_theta"],
        device=device,
        dtype=torch.float32,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    special_tokens = args.special_tokens or ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens,
    )
    text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
        special_tokens[0] if special_tokens else None,
    )
    print(text)


if __name__ == "__main__":
    main()
