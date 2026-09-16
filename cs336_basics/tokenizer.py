import json
import os

import regex as re
from collections.abc import Iterable


def _bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    offset = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, (chr(code_point) for code_point in code_points)))


class BPETokenizer:
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """Construct a tokenizer from GPT-2-style vocabulary and merge files."""
        byte_decoder = {character: byte for byte, character in _bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        vocab: dict[int, bytes] = {}
        for key, value in serialized_vocab.items():
            if isinstance(value, int):
                token_text, token_id = key, value
            else:
                token_text, token_id = value, int(key)
            vocab[token_id] = bytes(byte_decoder[character] for character in token_text)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                line = line.rstrip("\r\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes(byte_decoder[character] for character in left),
                        bytes(byte_decoder[character] for character in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}

        self.merges = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if not self.special_regex:
            return self._encode_text_segment(text)

        tokens = []

        last_pos = 0

        for match in self.special_regex.finditer(text):
            # " hello <|endoftext|> world"
            pre_text = text[last_pos : match.start()]

            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
                # pre_tokens : [1,2,3,...] self._encode_text_segment: [4,5,6] tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]

            special_tok = match.group()

            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])

            last_pos = match.end()

        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)

        for p_tok in pre_tokens:
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            while len(byte_parts) >= 2:
                best_pair = None
                min_rank = float("inf")

                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i + 1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break

                new_byte_parts = []
                i = 0
                # [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                while i < len(byte_parts):
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i + 1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts

            for part in byte_parts:
                ids.append(self.byte_to_id[part])

        return ids

    def decode(self, ids: list[int]) -> str:
        byte_segments = [self.id_to_byte[i] for i in ids]

        full_bytes = b"".join(byte_segments)

        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
