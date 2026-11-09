#!/usr/bin/env bash
# Keep only the newest complete checkpoint of each round-2 stage, so the
# full-finetune (~116 GB per CFG checkpoint) and LoRA (~33 GB) runs fit on the
# shared disk. Older ones are deleted once the newest has had no writes for
# 15 minutes (save finished). Login node, in the background:
#   nohup setsid bash slurm/recap_prune_ckpts.sh > logs/recap_prune.out 2>&1 < /dev/null &
REPO_DIR="${REPO_DIR:-$HOME/RECAP/openpi/RLinf}"
ROUNDS="${RECAP_PRUNE_ROUNDS:-round_002 round_002_full}"
# Checkpoints never deleted (<round>/<stage>/global_step_<N>), on top of the newest.
KEEP="${RECAP_PRUNE_KEEP:-round_002/cfg/global_step_25000 round_002_full/cfg/global_step_20000}"
while true; do
  for round in $ROUNDS; do
    for dir in "$REPO_DIR/recap_runs/$round"/{value/piper_recap_value,cfg/piper_recap_cfg}/checkpoints; do
      [[ -d "$dir" ]] || continue
      mapfile -t ckpts < <(find "$dir" -maxdepth 1 -name 'global_step_*' | sort -V)
      (( ${#ckpts[@]} > 1 )) || continue
      newest="${ckpts[-1]}"
      [[ -z "$(find "$newest" -newermt '-15 minutes' -print -quit)" ]] || continue
      stage=$(basename "$(dirname "$(dirname "$dir")")")
      for old in "${ckpts[@]:0:${#ckpts[@]}-1}"; do
        [[ " $KEEP " == *" $round/$stage/$(basename "$old") "* ]] && continue
        echo "$(date '+%F %T') delete $old (newest $(basename "$newest"))"
        rm -rf "$old"
      done
    done
  done
  sleep 600
done
