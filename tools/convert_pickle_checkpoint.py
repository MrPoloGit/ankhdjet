"""Convert a HuggingFace checkpoint stored as PyTorch pickle
(pytorch_model.bin) into a local safetensors directory ankhdjet's
frontend can load via `load_weights(local_dir)`.

Isolated and explicit on purpose: this is the ONE place in the
toolchain that reads pickle. `torch.load(..., weights_only=True)`
restricts unpickling to tensor data (PyTorch's own allowlisted-safe
path) -- it is not a full sandbox, so only point this at checkpoints
from sources you trust, same as you would before running any other
code against an untrusted .bin file.
`ankhdjet.frontend.hf.load_weights`/`load_config` never read pickle;
they only ever read this script's safetensors output. See
docs/binary_and_matmulfree_investigation.md for why this exists (the
FBI-LLM releases ship pickle, not safetensors).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SIDECAR_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
)


def convert(src_dir: Path, dst_dir: Path,
           bin_name: str = "pytorch_model.bin") -> Path:
    import torch
    from safetensors.torch import save_file

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(src_dir / bin_name, map_location="cpu",
                            weights_only=True)
    tensors = {k: v.contiguous() for k, v in state_dict.items()}
    out = dst_dir / "model.safetensors"
    save_file(tensors, out)

    for name in SIDECAR_FILES:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="directory containing pytorch_model.bin + config.json")
    ap.add_argument("dst", help="output directory for the safetensors conversion")
    ap.add_argument("--bin-name", default="pytorch_model.bin")
    args = ap.parse_args()
    out = convert(Path(args.src), Path(args.dst), args.bin_name)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
