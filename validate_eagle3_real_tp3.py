#!/usr/bin/env python3
"""
HARD proof: build REAL vLLM QKVParallelLinear + RowParallelLinear with an actual
tp_size=3 tensor-parallel group (single process, fake distributed), and run the
real per-shard weight loader against the PADDED q/k/v weights. This is the exact
code path the eagle3 draft hits at construction + weight-load.

If 64-head config were used here it would raise on divide(); with the padded 96
config it must succeed and accept out-dim 12288.
"""
import json
import os
import sys

import torch

DST = sys.argv[1] if len(sys.argv) > 1 else \
    "/cache/huggingface/MiniMax-M3-EAGLE3-pad96"
RANK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TP = 3
HEAD_DIM = 128

cfg = json.load(open(f"{DST}/config.json"))
NH = cfg["num_attention_heads"]
NKV = cfg["num_key_value_heads"]
QKV_IN = 2 * cfg["hidden_size"]   # eagle3 layer0 = 12288

# --- spin up a real TP group of size 3 (gloo, single proc acts as rank RANK) ---
# We do NOT need a real torch.distributed group to exercise the construction +
# weight-load math. The linear layers read tp world-size/rank purely through the
# getters below; divide() (the thing that raises on 64%3) reads the world size.
# Patch those getters to emulate tp_size=3 rank=RANK. This drives the EXACT
# divide()/shard-offset code paths the eagle3 draft hits at TP=3.
_ws = lambda: TP
_rk = lambda: RANK
# Patch the getters EVERYWHERE they were imported (each module binds its own ref).
import vllm.distributed.parallel_state as ps
ps.get_tensor_model_parallel_world_size = _ws
ps.get_tensor_model_parallel_rank = _rk
import vllm.model_executor.layers.linear as linmod
linmod.get_tensor_model_parallel_world_size = _ws
linmod.get_tensor_model_parallel_rank = _rk
import vllm.model_executor.parameter as parammod
parammod.get_tensor_model_parallel_rank = _rk
if hasattr(parammod, "get_tensor_model_parallel_world_size"):
    parammod.get_tensor_model_parallel_world_size = _ws

from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from safetensors import safe_open

print(f"=== build REAL QKVParallelLinear tp_size={TP} rank={RANK} "
      f"heads={NH} kv={NKV} head_dim={HEAD_DIM} qkv_in={QKV_IN} ===")
qkv = QKVParallelLinear(
    hidden_size=QKV_IN,
    head_size=HEAD_DIM,
    total_num_heads=NH,
    total_num_kv_heads=NKV,
    bias=False,
)
print(f"  qkv.num_heads(per part)={qkv.num_heads} num_kv_heads={qkv.num_kv_heads} "
      f"output_size_per_partition={qkv.output_size_per_partition}")
print(f"  qkv weight param shape={tuple(qkv.weight.shape)}")

o = RowParallelLinear(
    input_size=NH * HEAD_DIM,   # 12288
    output_size=cfg["hidden_size"],
    bias=False,
    input_is_parallel=True,
)
print(f"  o_proj.input_size_per_partition={o.input_size_per_partition} "
      f"weight shape={tuple(o.weight.shape)}")

# --- run the REAL per-shard weight loaders on the PADDED weights ---
print("\n=== run real per-shard weight loaders on padded q/k/v/o ===")
errs = []
with safe_open(f"{DST}/model.safetensors", framework="pt", device="cpu") as f:
    qw = f.get_tensor("layers.0.self_attn.q_proj.weight")
    kw = f.get_tensor("layers.0.self_attn.k_proj.weight")
    vw = f.get_tensor("layers.0.self_attn.v_proj.weight")
    ow = f.get_tensor("layers.0.self_attn.o_proj.weight")

for sid, w, nm in (("q", qw, "q_proj"), ("k", kw, "k_proj"), ("v", vw, "v_proj")):
    try:
        qkv.weight_loader(qkv.weight, w, sid)
        print(f"  [PASS] loaded {nm} shard '{sid}' out-dim {tuple(w.shape)} into qkv rank {RANK}")
    except Exception as e:
        errs.append((nm, repr(e)))
        print(f"  [FAIL] {nm}: {e!r}")

try:
    o.weight_loader(o.weight, ow)
    print(f"  [PASS] loaded o_proj in-dim {tuple(ow.shape)} into RowParallel rank {RANK}")
except Exception as e:
    errs.append(("o_proj", repr(e)))
    print(f"  [FAIL] o_proj: {e!r}")

# sanity: the qkv param fully populated after q+k+v shards for this rank
print(f"\n  fused qkv weight nonzero rows this rank: "
      f"{int(torch.count_nonzero(qkv.weight.sum(dim=1)))}/{qkv.weight.shape[0]}")

print("\n" + ("REAL-MODULE TP=3 LOAD: ALL PASS" if not errs else f"FAILURES: {errs}"))
sys.exit(0 if not errs else 1)
