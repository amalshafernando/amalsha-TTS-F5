#!/bin/bash
# run_finetune_sinhala.sh
# Full finetune of F5-TTS with a new Sinhala vocab (Option A)
# Run from repo root: bash run_finetune_sinhala.sh

set -e

DATASET_NAME="Sinhala_char_custom"
VOCAB_PATH="data/${DATASET_NAME}/vocab.txt"
CKPT_DIR="ckpts/${DATASET_NAME}"

echo "=== Step 1: Prepare dataset ==="
python src/f5_tts/train/datasets/prepare_sinhala.py

echo "=== Step 2: Setup accelerate (skip if already configured) ==="
# accelerate config   # uncomment to reconfigure

echo "=== Step 3: Launch finetuning ==="
accelerate launch --mixed_precision=fp16 \
    src/f5_tts/train/finetune_cli.py \
    --exp_name        F5TTS_v1_Base \
    --dataset_name    ${DATASET_NAME} \
    --tokenizer       custom \
    --tokenizer_path  ${VOCAB_PATH} \
    --finetune \
    --learning_rate       1e-5 \
    --batch_size_per_gpu  1600 \
    --batch_size_type     frame \
    --max_samples         64 \
    --epochs              200 \
    --num_warmup_updates  500 \
    --save_per_updates    2000 \
    --last_per_updates    1000 \
    --keep_last_n_checkpoints 5 \
    --logger              tensorboard

echo "=== Done. Checkpoints in: ${CKPT_DIR} ==="