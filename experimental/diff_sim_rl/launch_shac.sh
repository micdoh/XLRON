#!/bin/bash
# Launch a SHAC (or any train.py) run on a UCL GPU host inside tmux.
#
# Usage: ./launch_shac.sh <run_name> <gpu_uuid> [train.py flags...]
#
# - Pin GPUs by UUID, not index (a wedged GPU scrambles CUDA index order).
#   malmo usable H100s: GPU2=GPU-18355792-796a-023f-68f9-b2952eb378ee
#                       GPU3=GPU-fe12a3be-1f3a-122a-7229-312c32e5d08a
# - Uses the repo venv python directly (NOT uv run) so the shared NFS venv
#   is not re-synced under running jobs.
set -euo pipefail

NAME=$1; shift
GPU=$1; shift
REPO=${XLRON_REPO:-$HOME/git/xlron-mscl}
LOGDIR=${XLRON_LOGDIR:-$HOME/xlron-diffsim-logs}
mkdir -p "$LOGDIR"

CMD="cd $REPO && CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda \
  ./.venv/bin/python -m xlron.train.train $* 2>&1 | tee $LOGDIR/$NAME.log"

tmux new-session -d -s "$NAME" "bash -c '$CMD'"
echo "Launched tmux session '$NAME' on GPU $GPU. Log: $LOGDIR/$NAME.log"
