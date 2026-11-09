#!/usr/bin/env bash
# Move a running Piper RECAP train job (recap_round_train_8xa100.sbatch with
# --gres=gpu:4) to a whole 8x A100 node as soon as one is free. The job is
# cancelled only when the work lost since its last complete checkpoint is small
# (MAX_LOST_MIN); the 8-GPU job resumes from that checkpoint (DCP reshards
# 4 -> 8 ranks; global batches are fixed, only accumulation changes).
# Login node, in the background:
#   RECAP_SUBMIT_ENV=recap_runs/round_003/submit_env.sh nohup setsid \
#     bash slurm/recap_upgrade_to_8gpu.sh <train_jobid> >> logs/recap_upgrade.out 2>&1 < /dev/null &
# Writes the current train job id to $RUN_DIR/train_job.id; exits once the job
# runs on 8 GPUs, the round finishes, or the job disappears.
set -uo pipefail
REPO_DIR="${REPO_DIR:-$HOME/RECAP/openpi/RLinf}"
cd "$REPO_DIR" || exit 1
JOB="$1"
source "${RECAP_SUBMIT_ENV:?set RECAP_SUBMIT_ENV}"
RUN_DIR="$REPO_DIR/recap_runs/$RECAP_ROUND"
SCRIPT=slurm/recap_round_train_8xa100.sbatch
POLL="${POLL_SECONDS:-180}"
MAX_LOST_MIN="${MAX_LOST_MIN:-30}"
START_WAIT_MIN="${START_WAIT_MIN:-15}"
log() { echo "$(date '+%F %T') $*"; }
echo "$JOB" > "$RUN_DIR/train_job.id"

job_state() { squeue -j "$1" -h -o '%T' 2>/dev/null; }
job_gpus() { scontrol show job "$1" 2>/dev/null | grep -oP 'gres/gpu=\K[0-9]+' | head -1; }

# A node of the gpu partition with no GPU in use and room for the job's CPUs.
eight_free() {
  # The job asks for 64 CPUs and 800G (819200 MB) of memory.
  sinfo -h -N -p gpu -O "NodeList:12,StateLong:12,GresUsed:40,CPUsState:24,AllocMem:12,Memory:12" 2>/dev/null |
    awk '$2 ~ /^(idle|mixed)$/ && $3 ~ /gpu:\(null\):0\(/ {split($4,c,"/"); if (c[2] >= 64 && $6 - $5 >= 819200) {print $1; exit}}'
}

stage() {
  local s
  for s in value advantages cfg; do [[ -f "$RUN_DIR/$s.done" ]] || { echo "$s"; return; }; done
  echo done
}

# Seconds of work lost if the job stopped now; empty = do not stop now.
lost_seconds() {
  local st=$1 interval line cur spi ckpt step
  case $st in
    advantages) echo $(( $(date +%s) - $(date -r "$RUN_DIR/value.done" +%s) )); return ;;
    value) interval=1000 ;;
    cfg) interval=2500 ;;
    *) return ;;
  esac
  [[ -f "$RUN_DIR/$st.log" ]] || { echo 0; return; }   # still in micro-batch smoke tests
  line=$(tail -c 60000 "$RUN_DIR/$st.log" | tr '\r' '\n' | grep -a "Global Step" | tail -1)
  [[ -n "$line" ]] || { echo 0; return; }               # model still loading
  cur=$(grep -oP '\b\K[0-9]+(?=/[0-9]+ \[)' <<<"$line" | tail -1)
  spi=$(grep -oP '[0-9.]+(?=s/it)' <<<"$line" | tail -1)
  [[ -z "$spi" ]] && spi=$(grep -oP '[0-9.]+(?=it/s)' <<<"$line" | tail -1 | awk '{print 1/$1}')
  [[ -n "$cur" && -n "$spi" ]] || return
  # Never stop around a checkpoint save (last 3% before it, or < 3 min after).
  (( cur % interval > interval * 97 / 100 )) && return
  ckpt=$(find "$RUN_DIR/$st" -maxdepth 4 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)
  if [[ -n "$ckpt" ]]; then
    [[ -f "$ckpt/actor/model_state_dict/full_weights.pt" && -f "$ckpt/actor/dcp_checkpoint/.metadata" ]] || return
    [[ -z "$(find "$ckpt" -newermt '-3 minutes' -print -quit)" ]] || return
    step=${ckpt##*_}
  else
    step=0
  fi
  awk -v c="$cur" -v s="$step" -v t="$spi" 'BEGIN{printf "%d\n", (c-s)*t}'
}

log "watching job $JOB (round $RECAP_ROUND, ${MAX_LOST_MIN} min max lost work)"
while true; do
  state=$(job_state "$JOB")
  if [[ -z "$state" ]]; then log "job $JOB is gone (stage $(stage)); exiting"; exit 0; fi
  if [[ "$state" == RUNNING && "$(job_gpus "$JOB")" == 8 ]]; then log "job $JOB runs on 8 GPUs; done"; exit 0; fi
  st=$(stage)
  [[ "$st" == done ]] && { log "round finished"; exit 0; }
  node=$(eight_free)
  if [[ -n "$node" ]]; then
    lost=$(lost_seconds "$st")
    if [[ "$state" != RUNNING ]]; then lost=0; fi
    if [[ -n "$lost" ]] && (( lost <= MAX_LOST_MIN * 60 )); then
      log "8 free GPUs on $node; stage $st, ~$((lost / 60)) min lost; switching job $JOB"
      NEW=$(sbatch --parsable -p gpu --gres=gpu:8 --job-name=recap_train \
        --dependency=afterany:"$JOB" --export=ALL "$SCRIPT") || { log "sbatch failed"; sleep "$POLL"; continue; }
      scancel "$JOB"
      for _ in $(seq 1 $((START_WAIT_MIN * 2))); do
        sleep 30
        [[ "$(job_state "$NEW")" == RUNNING ]] && break
      done
      if [[ "$(job_state "$NEW")" == RUNNING ]]; then
        echo "$NEW" > "$RUN_DIR/train_job.id"
        log "job $NEW running on 8 GPUs ($(squeue -j "$NEW" -h -o %N)); done"
        exit 0
      fi
      log "job $NEW did not start within $START_WAIT_MIN min; back to 4 GPUs"
      scancel "$NEW"
      JOB=$(sbatch --parsable -p gpu --gres=gpu:4 --job-name=recap_train --export=ALL "$SCRIPT")
      echo "$JOB" > "$RUN_DIR/train_job.id"
      log "resubmitted 4-GPU job $JOB"
    else
      log "8 free GPUs on $node but stage $st would lose ${lost:-?}s (or a checkpoint save is near); waiting"
    fi
  fi
  sleep "$POLL"
done
