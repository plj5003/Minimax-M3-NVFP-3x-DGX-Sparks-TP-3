#!/bin/bash
# M3 TP=3 multi-node vLLM launcher, RoCE 200G data path (eugr mesh recipe).
# Runs INSIDE the vllm-m3-chthonic:nccl230u1 container (NCCL 2.30.7 w/ subnet-aware-routing).
# Usage: m3vllm-roce.sh leader   (Bluey/head 10.0.0.6)  |  m3vllm-roce.sh worker (Reddie/Asusi)
set -x
ROLE="${1:?usage: m3vllm-roce.sh leader|worker}"
HEAD_IP="${HEAD_IP:-10.0.0.6}"
RAY_PORT=6379
CLUSTER_GPUS=3

# --- b12x / M3 / arch envs (sm_121a for GB10) ---
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_MINIMAX_M3_MSA=1 VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_B12X_FP8_GEMM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=1
export B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export SAFETENSORS_FAST_GPU=1

# --- NCCL over RoCE (eugr 3-node mesh recipe). NCCL 2.30.7 (v2.30u1) provides SUBNET_AWARE_ROUTING. ---
# Ring legs (Agent B verified, live, GID idx3 RoCEv2): each node uses its two slot-1 RoCE HCAs
# (rocep1s0f0 + rocep1s0f1) to reach its two neighbors over distinct /30 subnets (192.168.100/101/102).
# Bootstrap/control stays on the 1GbE mgmt NIC (enP7s7); the DATA plane rides IB/RoCE.
# SUBNET_AWARE_ROUTING + MERGE_NICS=0 are what map each QP to the correct cable -> fixes the old err 110.
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=enP7s7 GLOO_SOCKET_IFNAME=enP7s7
export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_NICS=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_DEBUG=INFO
export HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0

SELF_IP="$(hostname -I | tr ' ' '\n' | grep -E '^10\.0\.0\.' | head -1)"
echo "ROLE=$ROLE SELF_IP=$SELF_IP HEAD_IP=$HEAD_IP"

# Ray (not in Luke's single-node image).
if ! python -c "import ray" 2>/dev/null; then
  pip install -q "ray==2.55.1" 2>&1 | tail -3 || pip install -q "ray[default]==2.55.1" 2>&1 | tail -3
fi

# b12x master (copy_runtime_metadata for chthonic).
if ! python -c "import inspect; from b12x.integration.paged_attention_scratch import B12XPagedAttentionScratchCaps as C; raise SystemExit(0 if 'copy_runtime_metadata' in inspect.signature(C.__init__).parameters else 1)" 2>/dev/null; then
  echo "Upgrading b12x -> master (08e980c)..."
  pip install -q --force-reinstall --no-deps git+https://github.com/lukealonso/b12x.git@08e980c303b0b6291700a6b85aa09aa874fc27cb 2>&1 | tail -3
fi

sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true

if [ "$ROLE" = "worker" ]; then
  for i in $(seq 1 60); do
    if ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=1 --node-ip-address="$SELF_IP" \
         --object-store-memory=1073741824 --block; then
      exit 0
    fi
    echo "worker: head not ready, retry in 5s..."; sleep 5
  done
  echo "worker: timed out joining head"; exit 1
fi

# leader
ray start --head --port="${RAY_PORT}" --num-gpus=1 --node-ip-address="${HEAD_IP}" --dashboard-host=0.0.0.0 \
  --object-store-memory=1073741824
echo "WAIT_FOR_${CLUSTER_GPUS}_GPU"
for i in $(seq 1 90); do
  if ray status 2>/dev/null | grep -qE "/${CLUSTER_GPUS}\.0 GPU"; then echo "RAY_CLUSTER_FULL_${CLUSTER_GPUS}_GPU"; break; fi
  sleep 5
done
ray status 2>&1 | tail -20

exec vllm serve lukealonso/MiniMax-M3-NVFP4 \
  --served-model-name minimax-m3 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 3 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.82 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend B12X_ATTN \
  --moe-backend b12x \
  -cc.mode=VLLM_COMPILE \
  -cc.cudagraph_mode=PIECEWISE \
  --block-size 128 \
  --load-format safetensors \
  --max-model-len 200000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --mm-encoder-tp-mode data \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
