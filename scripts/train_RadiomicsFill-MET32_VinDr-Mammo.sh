#!/bin/bash

export TRAIN_PATH="VinDr-Mammo_train_metadata.jsonl"
export VAL_PATH="VinDr-Mammo_validation_metadata.jsonl"
export PLOT_VAL_PATH="VinDr-Mammo_plotVal_metadata.jsonl"
export OUTPUT_DIR="/workspace/results/RadiomicsFill-MET32_VinDr-Mammo"

cd /workspace/source

CUDA_VISIBLE_DEVICES="0" accelerate launch train_RadiomicsFill-MET32_VinDr-Mammo.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
  --train_file=$TRAIN_PATH \
  --resolution=512 \
  --sample_weight_type="withRaiomics" \
  --image_column="file_name" \
  --otherSide_image_column="otherSide_file_name" \
  --caption_column="additional_feature" \
  --mixed_precision="fp16" \
  --train_batch_size=16 \
  --dataloader_num_workers=8 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --num_train_epochs=10000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-5 \
  --lr_scheduler="linear" \
  --lr_warmup_steps=10000 \
  --seed=42 \
  --plot_validation_file=$PLOT_VAL_PATH \
  --val_file=$VAL_PATH \
  --output_dir=$OUTPUT_DIR \
