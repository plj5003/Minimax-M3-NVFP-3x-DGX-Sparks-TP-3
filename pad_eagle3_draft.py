#!/usr/bin/env python3
"""
EAGLE3 draft-head padding surgery for TP=3 sharding.

Pads the Inferact/MiniMax-M3-EAGLE3 dense draft from
num_attention_heads=64 -> 96 by zero-filling 32 extra heads in bf16.

Why 96 (and not 66)? The target must satisfy BOTH constraints at once:
  - transformers config validation: hidden_size % num_heads == 0
      6144 / 96 == 64  (OK).   6144 / 66 -> fails: "hidden size (6144) is
      not a multiple of the number of attention heads (66)".
  - TP=3 divisibility: 96 % 3 == 0 (each rank gets 32 heads).
66 (the naive head_dim-based guess) passes TP divisibility but is rejected by
transformers because 6144 is not divisible by 66. 96 is the only value that
clears both walls, so it is the working target.

Run INSIDE the vllm_m3 container (has torch + safetensors).
Source draft is host-mounted at /cache/huggingface/... inside the container.

Padding plan (head_dim=128, 32 extra heads = 4096 rows/cols):
  q_proj.weight  out 8192 -> 12288  (rows; pad bottom with zeros)
  k_proj.weight  out 8192 -> 12288  (rows; kv heads 64->96)
  v_proj.weight  out 8192 -> 12288  (rows; kv heads 64->96)
  o_proj.weight  in  8192 -> 12288  (cols; pad right with zeros)
  everything else: byte-identical copy
  config.json: num_attention_heads=96, num_key_value_heads=96 (all else identical)
"""
import json
import os
import sys
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_DIM = 128
OLD_HEADS = 64
NEW_HEADS = 96
PAD_HEADS = NEW_HEADS - OLD_HEADS          # 32
OLD_PROJ = OLD_HEADS * HEAD_DIM            # 8192
NEW_PROJ = NEW_HEADS * HEAD_DIM            # 12288
PAD_ROWS = PAD_HEADS * HEAD_DIM            # 4096

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/cache/huggingface/hub/models--Inferact--MiniMax-M3-EAGLE3/snapshots/44cafa5ace418d8b22e2958df0c6aa1f2476842c"
DST = sys.argv[2] if len(sys.argv) > 2 else \
    "/cache/huggingface/MiniMax-M3-EAGLE3-pad96"

# Which tensors get padded and how.
PAD_OUT = {  # pad dim 0 (output rows): 8192 -> 12288
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
}
PAD_IN = {   # pad dim 1 (input cols): 8192 -> 12288
    "layers.0.self_attn.o_proj.weight",
}


def main():
    os.makedirs(DST, exist_ok=True)
    src_st = os.path.join(SRC, "model.safetensors")

    new_tensors = {}
    with safe_open(src_st, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k in PAD_OUT:
                assert t.shape[0] == OLD_PROJ, f"{k} out dim {t.shape[0]} != {OLD_PROJ}"
                pad = torch.zeros((PAD_ROWS, t.shape[1]), dtype=t.dtype)
                t2 = torch.cat([t, pad], dim=0)
                assert t2.shape[0] == NEW_PROJ
                new_tensors[k] = t2.contiguous()
                print(f"PAD-OUT  {k}: {tuple(t.shape)} -> {tuple(t2.shape)}")
            elif k in PAD_IN:
                assert t.shape[1] == OLD_PROJ, f"{k} in dim {t.shape[1]} != {OLD_PROJ}"
                pad = torch.zeros((t.shape[0], PAD_ROWS), dtype=t.dtype)
                t2 = torch.cat([t, pad], dim=1)
                assert t2.shape[1] == NEW_PROJ
                new_tensors[k] = t2.contiguous()
                print(f"PAD-IN   {k}: {tuple(t.shape)} -> {tuple(t2.shape)}")
            else:
                new_tensors[k] = t.contiguous()
                print(f"COPY     {k}: {tuple(t.shape)}")

    # keep safetensors metadata (preserves torchspec_version)
    save_file(new_tensors, os.path.join(DST, "model.safetensors"), metadata=meta)
    print(f"\nWrote {os.path.join(DST,'model.safetensors')}  metadata={meta}")

    # config.json: bump head counts, keep everything else identical
    with open(os.path.join(SRC, "config.json")) as cf:
        cfg = json.load(cf)
    assert cfg["num_attention_heads"] == OLD_HEADS
    assert cfg["num_key_value_heads"] == OLD_HEADS
    assert cfg["head_dim"] == HEAD_DIM
    assert cfg["hidden_size"] % NEW_HEADS == 0, \
        f"hidden_size {cfg['hidden_size']} not a multiple of {NEW_HEADS} (transformers would reject)"
    cfg["num_attention_heads"] = NEW_HEADS
    cfg["num_key_value_heads"] = NEW_HEADS
    with open(os.path.join(DST, "config.json"), "w") as cf:
        json.dump(cfg, cf, indent=2)
    print(f"Wrote config.json: num_attention_heads={NEW_HEADS} num_key_value_heads={NEW_HEADS}")

    # copy aux files (license/readme/gitattributes) for a clean self-contained dir
    for fn in os.listdir(SRC):
        if fn in ("model.safetensors", "config.json"):
            continue
        sp = os.path.join(SRC, fn)
        try:
            real = os.path.realpath(sp)
            if os.path.isfile(real):
                shutil.copy2(real, os.path.join(DST, fn))
                print(f"copied aux {fn}")
        except Exception as e:
            print(f"skip aux {fn}: {e}")

    print("\nDONE pad surgery.")


if __name__ == "__main__":
    main()
