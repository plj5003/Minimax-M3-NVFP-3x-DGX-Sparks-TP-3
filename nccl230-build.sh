#!/bin/bash
# Node-level driver: build NCCL v2.30u1 in a throwaway container, commit to vllm-m3-chthonic:nccl230u1.
# Does NOT touch the running vllm_m3 serving container. Logs to ~/nccl230-build.log, markers DONE/FAIL.
exec > ~/nccl230-build.log 2>&1
set -x
echo "START $(date)"
rm -f ~/nccl230-build.DONE ~/nccl230-build.FAIL
docker rm -f nccl_build 2>/dev/null || true
docker run -d --name nccl_build --network host --entrypoint sleep vllm-m3-chthonic:latest infinity || { touch ~/nccl230-build.FAIL; exit 1; }
docker cp ~/nccl230-inner-build.sh nccl_build:/inner-build.sh
if docker exec nccl_build bash /inner-build.sh; then
  echo "COMMITTING $(date)"
  docker commit nccl_build vllm-m3-chthonic:nccl230u1
  docker rm -f nccl_build
  echo "DONE $(date)"
  touch ~/nccl230-build.DONE
else
  echo "BUILD FAILED $(date)"
  docker rm -f nccl_build
  touch ~/nccl230-build.FAIL
fi
