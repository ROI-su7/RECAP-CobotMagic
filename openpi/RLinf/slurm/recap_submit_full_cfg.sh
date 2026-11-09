#!/usr/bin/env bash
# Independent whole-model RECAP pipeline for round 2 on A100, next to the LoRA
# run: Value (resumed from the LoRA run's newest Value checkpoint) ->
# advantages -> whole-model CFG, all in its own run dir and dataset copy.
# The copy hardlinks data/ and videos/ (read only) but has real meta/ files,
# because advantages write meta/advantages_*.parquet and mixture_config.yaml.
# Login node:
#   bash slurm/recap_submit_full_cfg.sh
# Full finetune on A100-40G: micro batch 1 peaks at 39.4 GB with full_shard,
# so the global batch is 64 (LoRA keeps 128); everything else matches LoRA.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/RECAP/openpi/RLinf}"
cd "$REPO_DIR"
SRC_RUN="recap_runs/${RECAP_SOURCE_ROUND:-round_002}"
DST_ROUND="${RECAP_FULL_ROUND:-round_002_full}"
SRC_DATA="recap_data/round_002"
DATA="$REPO_DIR/recap_data/$DST_ROUND"

if [[ ! -d "$DATA" ]]; then
  cp -al "$SRC_DATA" "$DATA"
  # Break the hardlinks of every meta file so writes stay in this copy.
  find "$DATA" -path '*/meta/*' -type f | while read -r f; do
    cp --remove-destination "$SRC_DATA/${f#"$DATA"/}" "$f"
  done
fi

RUN="recap_runs/$DST_ROUND"
CKPTS="value/piper_recap_value/checkpoints"
mkdir -p "$RUN/$CKPTS"
touch "$RUN/prep.done"
if [[ -z "$(ls "$RUN/$CKPTS")" ]]; then
  # Newest Value checkpoint that is complete (not written in the last 3 min).
  src=$(find "$SRC_RUN/$CKPTS" -maxdepth 1 -name 'global_step_*' -mmin +3 | sort -V | tail -1)
  cp -al "$src" "$RUN/$CKPTS/"
  echo "value resumes from $src" > "$RUN/SOURCE"
fi

export RECAP_ROUND="$DST_ROUND"
export RECAP_SFT_DATASET="$DATA/sft_all_wire_data" RECAP_DATASET="$DATA/lerobot_v2"
export RECAP_ROLLOUT_DATASETS="$DATA/lerobot_v1 $DATA/lerobot_v2"
export RECAP_FSDP_SHARDING=full_shard CFG_GLOBAL_BATCH=64 CFG_STEPS=30000
export CFG_MICRO_CANDIDATES="${CFG_MICRO_CANDIDATES:-2 1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 96% of 40 GB at mb=1
export CFG_EXTRA_ARGS="actor.model.recap_lora_only=false actor.model.recap_full_finetune=true actor.model.openpi.train_expert_only=false actor.fsdp_config.amp_autocast.enabled=true actor.fsdp_config.amp_autocast.precision=bf16 actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[GemmaMLP,SiglipEncoderLayer,SiglipVisionEmbeddings,GemmaRMSNorm,GemmaRotaryEmbedding]"
unset VALUE_MICRO_CANDIDATES RECAP_DEBUG_ANOMALY RECAP_RUN_DIR
source slurm/recap_server_env.sh
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage value
# hgx001: nvidia-smi hangs in D state (GPU driver fault, 2026-11-03).
job=$(sbatch --parsable -p "${RECAP_FULL_PARTITION:-gpu}" --job-name=recap_train_full \
  --exclude="${RECAP_FULL_EXCLUDE:-hgx001}" \
  --export=ALL slurm/recap_round_train_8xa100.sbatch)
echo "$(date '+%F %T') submitted $DST_ROUND job $job ($(cat "$RUN/SOURCE"))"
