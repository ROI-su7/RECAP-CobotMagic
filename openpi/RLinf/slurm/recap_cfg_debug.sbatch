#!/bin/bash
# A few CFG steps for debugging or memory checks (log: logs/recap_cfg_debug_<id>.out).
#   sbatch -p gpu slurm/recap_cfg_debug.sbatch
# Env: RECAP_DEBUG_ANOMALY (default 1; 0 for speed/memory checks), DEBUG_MB
# (micro batch, default 2), DEBUG_STEPS (default 3), CFG_EXTRA_ARGS (extra Hydra
# overrides, e.g. whole-model finetuning; README 9.8). Without advantage labels
# yet, add +data.plain_sft=true to CFG_EXTRA_ARGS.
#SBATCH --job-name=recap_cfg_debug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=logs/recap_cfg_debug_%j.out
#SBATCH --error=logs/recap_cfg_debug_%j.out

set -euo pipefail
REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/RECAP/openpi/RLinf}"
cd "$REPO_DIR"
source slurm/recap_server_env.sh
RAY_BIN="$(dirname "$RECAP_PYTHON")/ray"
export RLINF_NODE_RANK=0 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export RECAP_DEBUG_ANOMALY="${RECAP_DEBUG_ANOMALY:-1}"
[[ "$RECAP_DEBUG_ANOMALY" == 0 ]] && unset RECAP_DEBUG_ANOMALY
NGPU="${SLURM_GPUS_ON_NODE:-8}"
read -r -a CFG_EXTRA <<<"${CFG_EXTRA_ARGS:-}"
cleanup() { env -u RAY_ADDRESS "$RAY_BIN" stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup
mkdir -p "$RECAP_RUN_DIR"
# Sample GPU memory every 2 s for the memory check (killed with the job).
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 \
  >"$RECAP_RUN_DIR/debug_gpu_mem.txt" &
env -u RAY_ADDRESS "$RAY_BIN" start --head --port=6385 --node-ip-address=127.0.0.1 \
  --num-cpus=32 --num-gpus="$NGPU" --temp-dir="/tmp/ray_recap_dbg_${SLURM_JOB_ID}" --disable-usage-stats
"$RECAP_PYTHON" examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py \
  --config-path "$REPO_DIR/examples/offline_rl/config" --config-name piper_recap_cfg \
  "$CFG_DATA" data.num_workers=2 runner.max_epochs=1000000 \
  runner.logger.log_path="$RECAP_RUN_DIR/debug_cfg" \
  actor.micro_batch_size="${DEBUG_MB:-2}" actor.global_batch_size="$(( ${DEBUG_MB:-2} * NGPU ))" \
  runner.max_steps="${DEBUG_STEPS:-3}" runner.save_interval=1000000 "${CFG_EXTRA[@]}" \
  2>&1 | tee "$RECAP_RUN_DIR/debug_cfg.log"
echo "peak GPU memory (MiB): $(sort -n "$RECAP_RUN_DIR/debug_gpu_mem.txt" | tail -1)"
