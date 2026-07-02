#!/bin/bash
set -e

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

python -m transformer_lm.scripts.train \
  --train-tokens /lambda/nfs/LM-from-scratch/owt_train_50k.npy \
  --val-tokens   /lambda/nfs/LM-from-scratch/owt_val_50k.npy \
  --vocab-size   50000 \
  --model-size   small \
  --context-length 1024 \
  --batch-size   32 \
  --num-steps    27000 \
  --warmup-steps 1000 \
  --save-every   5000 \
  --log-every    100 \
  --val-every    500 \
  --alpha-max    1e-3 \
  --alpha-min    1e-4 \
  --flash-attention \
  --wandb \
  --wandb-project transformer-lm \
  --wandb-run    small-50k-flash \
  --out-dir      run-50k-small-flash
