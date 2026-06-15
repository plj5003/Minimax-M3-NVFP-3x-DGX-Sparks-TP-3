#!/usr/bin/env python3
"""
DRY validation of the padded EAGLE3 draft (no serving, no cluster, single proc).

Checks:
  1. padded safetensors shapes are self-consistent (q/k/v out=12288, o in=12288, rest unchanged)
  2. config has 96 heads / 96 kv, head_dim 128, 96 % 3 == 0, and 6144 % 96 == 0
  3. mimic vLLM head math (llama.py LlamaAttention) at tp_size=3 -> asserts pass
  4. instantiate vLLM QKVParallelLinear + RowParallelLinear at tp_size=3 (disable_tp=False
     emulation via direct divide math) to confirm shard sizes are integral and the
     per-shard q/k/v loader would accept the padded out-dim 12288.
"""
import json
import sys

import torch
from safetensors import safe_open
from vllm.distributed.utils import divide

HEAD_DIM = 128
TP = 3
NEW_HEADS = 96
NEW_PROJ = NEW_HEADS * HEAD_DIM  # 12288

DST = sys.argv[1] if len(sys.argv) > 1 else \
    "/cache/huggingface/MiniMax-M3-EAGLE3-pad96"

ok = True


def check(cond, msg):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {msg}")


# ---- 1 & shapes ----
shapes = {}
with safe_open(f"{DST}/model.safetensors", framework="pt", device="cpu") as f:
    meta = f.metadata()
    for k in f.keys():
        shapes[k] = tuple(f.get_tensor(k).shape)

print("=== padded tensor shapes ===")
for k in sorted(shapes):
    print(f"  {k}: {shapes[k]}")
print(f"  metadata={meta}\n")

check(shapes["layers.0.self_attn.q_proj.weight"] == (NEW_PROJ, 12288), "q_proj out=12288, in=12288")
check(shapes["layers.0.self_attn.k_proj.weight"] == (NEW_PROJ, 12288), "k_proj out=12288, in=12288")
check(shapes["layers.0.self_attn.v_proj.weight"] == (NEW_PROJ, 12288), "v_proj out=12288, in=12288")
check(shapes["layers.0.self_attn.o_proj.weight"] == (6144, NEW_PROJ), "o_proj out=6144, in=12288")
# untouched
check(shapes["embed_tokens.weight"] == (200064, 6144), "embed_tokens untouched")
check(shapes["lm_head.weight"] == (200064, 6144), "lm_head untouched")
check(shapes["fc.weight"] == (6144, 18432), "fc untouched")
check(shapes["norm.weight"] == (6144,), "norm untouched")
check(shapes["layers.0.input_layernorm.weight"] == (6144,), "input_layernorm untouched (hidden dim)")
check(shapes["layers.0.hidden_norm.weight"] == (6144,), "hidden_norm untouched (hidden dim)")

# verify the 32 extra heads are actually zero (so they contribute nothing)
with safe_open(f"{DST}/model.safetensors", framework="pt", device="cpu") as f:
    q = f.get_tensor("layers.0.self_attn.q_proj.weight")
    o = f.get_tensor("layers.0.self_attn.o_proj.weight")
check(torch.count_nonzero(q[8192:]) == 0, "q_proj padded rows (8192:12288) are all zero")
check(torch.count_nonzero(o[:, 8192:]) == 0, "o_proj padded cols (8192:12288) are all zero")

# ---- 2 config ----
cfg = json.load(open(f"{DST}/config.json"))
check(cfg["num_attention_heads"] == NEW_HEADS, f"config num_attention_heads={cfg['num_attention_heads']}")
check(cfg["num_key_value_heads"] == NEW_HEADS, f"config num_key_value_heads={cfg['num_key_value_heads']}")
check(cfg["head_dim"] == HEAD_DIM, f"config head_dim={cfg['head_dim']}")
check(NEW_HEADS % TP == 0, f"{NEW_HEADS} % {TP} == 0 (TP divisibility)")
check(cfg["hidden_size"] % NEW_HEADS == 0, f"{cfg['hidden_size']} % {NEW_HEADS} == 0 (transformers config validation)")

# ---- 3 mimic vLLM llama.py LlamaAttention head math at tp_size=3 ----
print("\n=== mimic vLLM LlamaAttention.__init__ head math at tp_size=3 ===")
total_num_heads = cfg["num_attention_heads"]
total_num_kv_heads = cfg["num_key_value_heads"]
try:
    assert total_num_heads % TP == 0          # llama.py line 145
    num_heads = total_num_heads // TP
    assert total_num_kv_heads % TP == 0       # llama.py line 151
    num_kv_heads = max(1, total_num_kv_heads // TP)
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    o_input = total_num_heads * HEAD_DIM      # llama.py line 176
    print(f"  num_heads/part={num_heads}  num_kv_heads/part={num_kv_heads}")
    print(f"  q_size/part={q_size}  kv_size/part={kv_size}  o_proj input_size={o_input}")
    check(True, "LlamaAttention asserts pass at tp_size=3")
    check(o_input == NEW_PROJ, "o_proj input_size matches padded in-dim 12288")
except AssertionError:
    check(False, "LlamaAttention asserts pass at tp_size=3")

# ---- 4 QKVParallelLinear divide math (linear.py) at tp_size=3 ----
print("\n=== QKVParallelLinear divide() math at tp_size=3 ===")
try:
    nh = divide(total_num_heads, TP)          # linear.py line 1034
    nkv = divide(total_num_kv_heads, TP)      # line 1039 (tp < kv heads)
    out_q = nh * HEAD_DIM * TP                 # reconstructed full q out
    out_k = nkv * HEAD_DIM * TP
    out_v = nkv * HEAD_DIM * TP
    print(f"  divide(64+32={total_num_heads},3)={nh}  q per-shard out={nh*HEAD_DIM}")
    print(f"  full q out reconstructed={out_q} (loader expects loaded_weight out-dim {out_q})")
    check(out_q == NEW_PROJ, "fused qkv 'q' shard expects out-dim 12288 == padded q_proj out")
    check(out_k == NEW_PROJ, "fused qkv 'k' shard expects out-dim 12288 == padded k_proj out")
    check(out_v == NEW_PROJ, "fused qkv 'v' shard expects out-dim 12288 == padded v_proj out")
    # per-shard slice sizes must be integral (divide raises if not)
    check(True, "divide() did not raise -> all shard sizes integral")
except Exception as e:
    check(False, f"QKVParallelLinear divide math: {e!r}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
