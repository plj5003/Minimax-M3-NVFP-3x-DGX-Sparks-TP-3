#!/bin/bash
# Runs INSIDE a throwaway container off vllm-m3-chthonic:latest.
# Builds NVIDIA NCCL v2.30u1 for sm_121 (GB10) and overrides the pip-bundled libnccl.so.2.
set -ex
cd /opt
rm -rf nccl230
git clone --depth 1 -b v2.30u1 https://github.com/NVIDIA/nccl.git nccl230
cd nccl230
nice -n 19 make -j12 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
ls -la build/lib/libnccl.so*
PIP=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib
cp -a "$PIP/libnccl.so.2" /opt/libnccl.so.2.BACKUP-229 2>/dev/null || true
rm -f "$PIP/libnccl.so.2"
ln -s /opt/nccl230/build/lib/libnccl.so.2 "$PIP/libnccl.so.2"
echo "=== VERIFY NCCL VERSION ==="
python -c 'import torch; print("NCCLVER", torch.cuda.nccl.version())'
echo "=== SUBNETAWARE CHECK ==="
strings /opt/nccl230/build/lib/libnccl.so.2 | grep -i SUBNET_AWARE | head
echo "INNER_DONE"
