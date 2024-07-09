"""
Adapted training script for RadiomicsFill-Mammo: Synthetic Mammogram Mass Manipulation with Radiomics Features.

Original source code adapted from:
 * Diffusers (http://www.apache.org/licenses/LICENSE-2.0)
   * https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
   * https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
 * Terrain Diffusion
   * https://github.com/sshh12/terrain-diffusion/blob/main/scripts/train_text_to_image_lora_sd2_inpaint.py

Example Usage (40GB VRAM, tested A100):
$ accelerate launch train_RadiomicsFill-MET32_VinDr-Mammo.py \
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
  --num_train_epochs=1000 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-5 \
  --lr_scheduler="linear" \
  --lr_warmup_steps=10000 \
  --seed=42 \
  --plot_validation_file=$PLOT_VAL_PATH \
  --val_file=$VAL_PATH \
  --output_dir=$OUTPUT_DIR

Example `train_file.jsonl`, `plot_validation_file.jsonl`, and `val_file.jsonl`:
    {"file_name": "0000_R_MLO.nii.gz", "mask_file_name": null, "otherSide_file_name": "0000_L_MLO_to_R_MLO.nii.gz", "additional_feature": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]}
    ...   
   
"""

import argparse
import logging
import math
import os
import pandas as pd
import random
import shutil
import json
from pathlib import Path
import matplotlib.pyplot as plt

from skimage.transform import resize
import SimpleITK as sitk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import Dataset, DataLoader
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from torch.utils.data import WeightedRandomSampler
from scipy.spatial.distance import cdist

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from Custom_pipline_stable_diffusion_inpaint_withOtherSide_MET import StableDiffusionInpaintPipelineWithOtherSideWithMET

import wandb

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, feature_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    val_transforms = VinDrMammoTransform(args.resolution, is_train=False, is_random_mask_expansion=True, current_epoch=epoch, is_plot_random_expansion=True)
    val_dataset = VinDrMammoDataset(args.plot_validation_file, transform=val_transforms, is_train=False)

    logger.info(
        f"Running plot validation... \n Generating {len(val_dataset)} images"
    )

    pipeline = StableDiffusionInpaintPipelineWithOtherSideWithMET.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        feature_encoder=accelerator.unwrap_model(feature_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    fig, axs = plt.subplots(len(val_dataset)*2, 5, figsize=(25, 10*len(val_dataset)))
    # For mass cases
    for v_i, val_example in enumerate(tqdm(val_dataset)):
        val_image = val_example["pixel_values"]
        mask_image = val_example["masks"]
        unmasked_idx = val_example["unmasked_idx"]
        unmasked_x = val_example["unmasked_x"]

        with torch.autocast("cuda"):
            result_image = pipeline(
                    unmasked_x = unmasked_x,
                    unmasked_idx = unmasked_idx,
                    image=val_image,
                    mask_image=mask_image,
                    num_inference_steps=999,
                    strength=1.0,
                    generator=generator,
                    output_type="np",
                ).images[0]

        val_image = val_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        masked_image = val_example["masked_images"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        roi_image = val_example["rois"]

        axs[v_i, 0].imshow(val_image[:,:,2], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 0].set_title("Other Side Image")
        axs[v_i, 0].axis('off')
        
        axs[v_i, 1].imshow(val_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 1].set_title("Original Image")
        axs[v_i, 1].axis('off')

        axs[v_i, 2].imshow(result_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 2].set_title("Result Image")
        axs[v_i, 2].axis('off')

        axs[v_i, 3].imshow(masked_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 3].set_title("Masked Image")
        axs[v_i, 3].axis('off')

        axs[v_i, 4].imshow(mask_image[0]+roi_image[0], cmap='gray', vmin=0, vmax=2)
        axs[v_i, 4].set_title("ROI Image")
        axs[v_i, 4].axis('off')

    # For normal cases
    for v_i, val_example in enumerate(tqdm(val_dataset), start=len(val_dataset)):
        val_image = val_example["pixel_values"]
        mask_image = val_example["masks"]
        unmasked_idx = val_example["unmasked_idx"]
        unmasked_x = val_example["unmasked_x"]
        unmasked_x[:,:-2] = 0
        unmasked_x[:,-1] = 0
        with torch.autocast("cuda"):
            result_image = pipeline(
                    unmasked_x = unmasked_x,
                    unmasked_idx = unmasked_idx,
                    image=val_image,
                    mask_image=mask_image,
                    num_inference_steps=999,
                    strength=1.0,
                    generator=generator,
                    output_type="np",
                ).images[0]

        val_image = val_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        masked_image = val_example["masked_images"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        roi_image = val_example["rois"]

        axs[v_i, 0].imshow(val_image[:,:,2], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 0].set_title("Other Side Image")
        axs[v_i, 0].axis('off')
        
        axs[v_i, 1].imshow(val_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 1].set_title("Original Image")
        axs[v_i, 1].axis('off')

        axs[v_i, 2].imshow(result_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 2].set_title("Result Image")
        axs[v_i, 2].axis('off')

        axs[v_i, 3].imshow(masked_image[:,:,0], cmap='gray', vmin=0, vmax=1)
        axs[v_i, 3].set_title("Masked Image")
        axs[v_i, 3].axis('off')

        axs[v_i, 4].imshow(mask_image[0]+roi_image[0], cmap='gray', vmin=0, vmax=2)
        axs[v_i, 4].set_title("ROI Image")
        axs[v_i, 4].axis('off')

    if not os.path.isdir(os.path.join(args.output_dir, "validation_plots")):
        os.mkdir(os.path.join(args.output_dir, "validation_plots"))
    
    # save figure
    plt.savefig(os.path.join(args.output_dir, "validation_plots", f"validation_example_{epoch}.png"), dpi=300)
    plt.close("all")
    del pipeline
    torch.cuda.empty_cache()

    return

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sample_weight_type",
        type=str,
        default=None,
        help="sample_weight_type [None, 'withRaiomics', 'onlyClass']",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help=(
            "A .jsonl file containing paths to images and their corresponding prompts for training."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="file_name",
        help="The key in the .jsonl file representing the image path.",
    )
    parser.add_argument(
        "--mask_column",
        type=str,
        default="mask_file_name",
        help="The key in the .jsonl file representing the mask image path.",
    )
    parser.add_argument(
        "--otherSide_image_column",
        type=str,
        default="otherSide_file_name",
        help="The key in the .jsonl file representing the registered other side image path.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="additional_feature",
        help="The key in the .jsonl file representing the condition feature values (radiomics features, density, BIRADS) array.",
    )
    parser.add_argument(
        "--plot_validation_file",
        type=str,
        default=None,
        help="A .jsonl file containing paths to images and their corresponding prompts for plotting validation results.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="A .jsonl file containing paths to images and their corresponding prompts for validation.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--random_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_vflip",
        action="store_true",
        help="whether to randomly flip images vertically",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="RadiomicsFill-Mammo",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args

args = parse_args()

class VinDrMammoTransform:
    def __init__(self, resolution, is_train=True, is_random_mask_expansion=True, current_epoch=0, is_plot_random_expansion=False):
        self.resolution = resolution
        self.is_train = is_train
        self.current_epoch = current_epoch
        self.is_random_mask_expansion = is_random_mask_expansion
        self.is_plot_random_expansion = is_plot_random_expansion

    def resize_half(self, image, mask):
        image_resized = F.interpolate(image, scale_factor=1/3, mode='bilinear', align_corners=False)
        if mask is not None:
            mask_resized = F.interpolate(mask, scale_factor=1/3, mode='nearest')
        else:
            mask_resized = None
        return image_resized, mask_resized
    
    def pad_to_resolution(self, image, mask, bbox):
        y, x = image.shape[-2:]
        pad_y = max(0, self.resolution - y)
        pad_x = max(0, self.resolution - x)
        # Apply padding
        image_padded = F.pad(image, (0, pad_x, 0, pad_y), "constant", 0)
        mask_padded = F.pad(mask, (0, pad_x, 0, pad_y), "constant", 0)
        bbox_padded = F.pad(bbox, (0, pad_x, 0, pad_y), "constant", 0)
        return image_padded, mask_padded, bbox_padded
    
    def get_expanded_bbox(self, image, mask):
        y, x = image.shape[-2:]
        bbox = torch.zeros_like(mask)
        mask_indices = torch.nonzero(mask>0.5, as_tuple=False)

        if len(mask_indices) > 0:
            mask_y, mask_x = mask_indices[:, 2], mask_indices[:, 3]
            min_y, max_y = torch.min(mask_y), torch.max(mask_y)
            min_x, max_x = torch.min(mask_x), torch.max(mask_x)

            bbox[:, :, min_y:max_y, min_x:max_x] = 1
        return bbox
    
    def random_crop_flip(self, image, roi, bbox):
        y, x = image.shape[-2:]
        mask_indices = torch.nonzero(bbox > 0.5, as_tuple=False)

        if len(mask_indices) > 0:
            mask_y, mask_x = mask_indices[:, 2], mask_indices[:, 3]
            min_y, max_y = torch.min(mask_y), torch.max(mask_y)
            min_x, max_x = torch.min(mask_x), torch.max(mask_x)
            center_y, center_x = (min_y + max_y) // 2, (min_x + max_x) // 2

            # Check the height and width of the mask
            mask_height = max_y - min_y
            mask_width = max_x - min_x

            if self.is_train:
                if mask_height < self.resolution:
                    vertical_margin = max(0, y - self.resolution)  # Maximum margin for vertical cropping
                    top = random.randint(max(0, min_y - (self.resolution - mask_height)), 
                                        min(min_y, vertical_margin))
                else:
                    top = max(0, min(center_y - self.resolution // 2, y - self.resolution))

                if mask_width < self.resolution:
                    horizontal_margin = max(0, x - self.resolution)  # Maximum margin for horizontal cropping
                    left = random.randint(max(0, min_x - (self.resolution - mask_width)), 
                                        min(min_x, horizontal_margin))
                else:
                    left = max(0, min(center_x - self.resolution // 2, x - self.resolution))

            else:
                # In test mode, center the crop around the mask's center
                top = int(max(0, min(y - self.resolution, center_y - self.resolution // 2)))
                left = int(max(0, min(x - self.resolution, center_x - self.resolution // 2)))
                        
        else:
            # If there is no region with a value of 1 in the mask, perform a general random crop
            top = random.randint(0, max(0, y - self.resolution))
            left = random.randint(0, max(0, x - self.resolution))

        image_cropped = image[:, :, top:top+self.resolution, left:left+self.resolution]
        roi_cropped = roi[:, :, top:top+self.resolution, left:left+self.resolution]
        bbox_cropped = bbox[:, :, top:top+self.resolution, left:left+self.resolution]

        bg_mask = image_cropped[:, 0:1, :, :] == 0
        bbox_cropped[bg_mask] = 0

        if self.is_train:
            # Apply random flips
            if args.random_hflip and random.random() > 0.5:
                image_cropped = torch.flip(image_cropped, [2])
                mask_cropped = torch.flip(mask_cropped, [2])
                bbox_cropped = torch.flip(bbox_cropped, [2])
            if args.random_vflip and random.random() > 0.5:
                image_cropped = torch.flip(image_cropped, [3])
                mask_cropped = torch.flip(mask_cropped, [3])
                bbox_cropped = torch.flip(bbox_cropped, [3])

        return image_cropped, roi_cropped, bbox_cropped
    
    def normalize(self, image):
        # Assuming image values are already between 0 and 1, so just centering around 0
        return (image - 0.5) / 0.5
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_random_box_mask(self, image):
        y, x = image.shape[-2:]
        mask = torch.zeros((1, 1, y, x))

        top = random.randint(0, y)
        left = random.randint(0, x)
        height = random.randint(0, y - top)
        width = random.randint(0, x - left)

        mask[:, :, top:top+height, left:left+width] = 1

        return mask

    def __call__(self, image, mask):
        image_resized, mask_resized = self.resize_half(image, mask)
        if mask_resized is not None:
            bbox_resized = self.get_expanded_bbox(image_resized, mask_resized)
        else:
            bbox_resized = self.get_random_box_mask(image_resized)
            mask_resized = bbox_resized
        image_padded, mask_padded, bbox_padded = self.pad_to_resolution(image_resized, mask_resized, bbox_resized)

        image_transformed, mask_transformed, bbox_transformed = self.random_crop_flip(image_padded, mask_padded, bbox_padded)

        if self.is_train:
            image_transformed = self.normalize(image_transformed)

        return image_transformed, mask_transformed, bbox_transformed


class VinDrMammoDataset(Dataset):
    def __init__(self, jsonl_path, transform=None, is_train=True, current_epoch=0):
        self.transform = transform
        self.is_train = is_train
        self.data = []
        self.current_epoch = current_epoch
        if self.transform:
            self.transform.set_epoch(self.current_epoch)
        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def updata_epoch(self, epoch):
        self.current_epoch = epoch
        if self.transform:
            self.transform.set_epoch(self.current_epoch)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        file_name = item[args.image_column]
        mask_file_name = item[args.mask_column]
        otherSide_file_name = item[args.otherSide_image_column]

        prompt = item[args.caption_column]     

        image = sitk.GetArrayFromImage(sitk.ReadImage(file_name))

        if mask_file_name is not None:
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_file_name)).astype(np.uint8)
        else:
            mask = None
        otherSide_image = sitk.GetArrayFromImage(sitk.ReadImage(otherSide_file_name))

        otherSide_image = (otherSide_image - otherSide_image.min()) / (otherSide_image.max() - otherSide_image.min())

        image = np.stack([image[0], image[0], otherSide_image], axis=0)

        image = torch.tensor(image).unsqueeze(0)
        if mask is not None:
            mask = torch.tensor(mask).unsqueeze(0)
            bbox = torch.zeros_like(mask)

        if self.transform:
            image, mask, bbox = self.transform(image, mask)

        image = image.squeeze(0)
        mask = mask.squeeze(0).to(torch.uint8)
        bbox = bbox.squeeze(0).to(torch.uint8)

        masked_images = torch.cat([
            image[0:1] * (bbox < 0.5),  # Apply mask to the first channel
            image[1:2] * (bbox < 0.5),  # Apply mask to the second channel
            image[2:3]  # Keep the third channel unchanged
        ], dim=0)  # Concatenate tensors along the channel dimension

        unmasked_x = torch.tensor(prompt, dtype=torch.float32)
        unmasked_idx = torch.arange(0, unmasked_x.size(0), dtype=torch.long)

        if not self.is_train:
            return {
                "pixel_values": image,
                "unmasked_x": unmasked_x.unsqueeze(0),
                "unmasked_idx": unmasked_idx.unsqueeze(0),
                "masks": bbox,
                "rois": mask,
                "masked_images": masked_images,
            }
        
        image = torch.stack([image[0], image[0], image[0]])

        return {
            "pixel_values": image,
            "unmasked_x": unmasked_x,
            "unmasked_idx": unmasked_idx,
            "masks": bbox,
            "masked_images": masked_images,
        }
    
class MET(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 69,
        embedding_dim: int = 64,
        n_head: int = 1,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        dtype=torch.float
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.dtype = dtype

        # Subtract 1 from desired embedding dim to account for token
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim - 1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, 
            nhead=n_head, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.transformer_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Flatten())
        self.sigmoid = nn.Sigmoid()

        self.mask_embed_layer = nn.Linear(1,1)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights"""
        factor = 1.0
    
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.weight.data.normal_(mean=0.0, std=factor * 0.02)
                module.bias.data.zero_()

    def embed_inputs(self, x, idx):
        embd = self.embedding(idx)
        return torch.cat([x.unsqueeze(-1), embd], dim=-1)

    def forward(self, unmasked_x, unmasked_idx, masked_x, masked_idx):
        batch_size = masked_x.size(0)
        fixed_input = torch.ones(batch_size, 1).to(device)
        mask_embed = self.mask_embed_layer(fixed_input)
        seq_len = masked_x.size(1)
        mask_embed = mask_embed.repeat(1, seq_len)
         
        unmasked_inputs = self.embed_inputs(unmasked_x, unmasked_idx)
        masked_inputs = self.embed_inputs(mask_embed, masked_idx)
        
        # Input unmasked_inputs to the encoder
        encoder_output = self.transformer_encoder(unmasked_inputs)
        # Combine the encoder output with masked_inputs
        decoder_input = torch.concat([encoder_output, masked_inputs], dim=1)
        # Input decoder_input to the decoder
        decoder_output = self.transformer_decoder(tgt=decoder_input, memory=encoder_output)

        x_hat = self.transformer_head(decoder_output)
        x_hat = self.sigmoid(x_hat)

        last_hidden_state = self.final_layer_norm(encoder_output)
        pooled_output = last_hidden_state[:,-1]

        return x_hat, pooled_output
    
    def encode(self, x, idx):
        inputs = self.embed_inputs(x, idx)
        return self.transformer_encoder(inputs)
    
def main():
    import wandb
    wandb.init(project="RadiomicsFill-Mammo")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
        
        model_root = '/workspace/source/saved_model'
        model_name = 'MET_VinDr-Mammo_Mass_enc6_dec3_head1_embedding32_maxpmask90_best_val.pth'
        model_path = os.path.join(model_root, model_name)

        embedding_dim = 32
        num_encoder_layers = 6
        num_decoder_layers = 3
        num_head = 1

        feature_encoder = MET(num_embeddings=69, embedding_dim=embedding_dim, n_head = num_head, num_encoder_layers = num_encoder_layers,
                    num_decoder_layers = num_decoder_layers, dim_feedforward = 64, dropout = 0.1)
        feature_encoder.load_state_dict(torch.load(model_path))

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", cross_attention_dim=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    feature_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_transforms = VinDrMammoTransform(args.resolution, is_train=True, is_random_mask_expansion=True)
    train_dataset = VinDrMammoDataset(args.train_file, transform=train_transforms, is_train=True)

    if args.sample_weight_type is not None:

        samples_normal = pd.read_csv('/workspace/source/normal_train_prompts.csv')
        samples_mass = pd.read_csv('/workspace/source/mass_train_prompts.csv')
        
        # Calculate cosine distances for samples_mass
        distances_mass = cdist(samples_mass, samples_mass, metric='cosine')
        np.fill_diagonal(distances_mass, np.inf)
        min_distances_mass = np.min(distances_mass, axis=1)
        epsilon = 1e-5
        rarity_scores_mass = 1 / (min_distances_mass + epsilon)
        weights_mass = rarity_scores_mass / rarity_scores_mass.sum()

        # Set the weights for the normal group to 1/10 of the smallest weight in the mass group
        min_weight_mass = np.min(weights_mass)
        weights_normal = np.full(len(samples_normal), min_weight_mass / 2)

        # Create the full weight array
        weights = np.concatenate([weights_mass, weights_normal])

        # Normalize the full weight array
        weights /= weights.sum()

        # Print the results
        print("Training weights", len(weights), weights)
        print(weights[0], weights[-1])

        # Create a tensor for weights and set up the WeightedRandomSampler
        weights_tensor = torch.FloatTensor(weights)
        # Create the WeightedRandomSampler
        sampler = WeightedRandomSampler(weights_tensor, num_samples=len(samples_mass) * 2, replacement=True)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            # shuffle=True,
            sampler=sampler,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            # sampler=sampler,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    val_transforms = VinDrMammoTransform(args.resolution, is_train=False, is_random_mask_expansion=True)
    val_dataset = VinDrMammoDataset(args.val_file, transform=val_transforms, is_train=True)

    if args.sample_weight_type is not None:

        samples_normal = pd.read_csv('/workspace/source/normal_val_prompts.csv')
        samples_mass = pd.read_csv('/workspace/source/mass_val_prompts.csv')
        
        # Calculate cosine distances for samples_mass
        distances_mass = cdist(samples_mass, samples_mass, metric='cosine')
        np.fill_diagonal(distances_mass, np.inf)
        min_distances_mass = np.min(distances_mass, axis=1)
        epsilon = 1e-5
        rarity_scores_mass = 1 / (min_distances_mass + epsilon)
        weights_mass = rarity_scores_mass / rarity_scores_mass.sum()

        # Set the weights for the normal group to 1/10 of the smallest weight in the mass group
        min_weight_mass = np.min(weights_mass)
        weights_normal = np.full(len(samples_normal), min_weight_mass / 2)

        # Create the full weight array
        weights = np.concatenate([weights_mass, weights_normal])

        # Normalize the full weight array
        weights /= weights.sum()

        # Print the results
        print("Validation weights", len(weights), weights)
        print(weights[0], weights[-1])

        # Create a tensor for weights and set up the WeightedRandomSampler
        weights_tensor = torch.FloatTensor(weights)
        # Create the WeightedRandomSampler
        val_sampler = WeightedRandomSampler(weights_tensor, num_samples=len(samples_mass), replacement=True)

        # DataLoaders creation:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            # shuffle=False,
            sampler=val_sampler,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    else:

        # DataLoaders creation:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    feature_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    best_val_loss = 999999

    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataset.updata_epoch(epoch)
        val_dataset.updata_epoch(epoch)

        unet.train()

        epoch_train_loss = 0.0
        epoch_val_loss_300_500_700 = [0.0, 0.0, 0.0]

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # if args.checkpoints_total_limit is not None:
                        #     checkpoints = os.listdir(args.output_dir)
                        #     checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        #     if len(checkpoints) >= args.checkpoints_total_limit:
                        #         num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        #         removing_checkpoints = checkpoints[0:num_to_remove]

                        #         logger.info(
                        #             f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        #         )
                        #         logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        #         for removing_checkpoint in removing_checkpoints:
                        #             removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        #             shutil.rmtree(removing_checkpoint)

                        # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")

                        if args.plot_validation_file is not None:
                            log_validation(
                                vae,
                                feature_encoder,
                                tokenizer,
                                unet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                            )

                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                unmasked_x = batch["unmasked_x"].to(dtype=weight_dtype, device=latents.device)
                unmasked_idx = batch["unmasked_idx"].to(dtype=torch.long, device=latents.device)
                
                encoder_hidden_states = feature_encoder.encode(unmasked_x, unmasked_idx)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                masked_image_latents = vae.encode(
                    batch["masked_images"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                mask = torch.nn.functional.interpolate(
                    batch["masks"],
                    size=(
                        args.resolution // vae_scale_factor,
                        args.resolution // vae_scale_factor,
                    ),
                )
                mask = mask.to(device=latents.device, dtype=weight_dtype)

                latent_model_input = torch.cat(
                    [noisy_latents, mask, masked_image_latents], dim=1
                )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                        
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()
                epoch_train_loss += train_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        unet.eval()
        # Caculate Validation loss

        logger.info("***** Running validation *****")
        logger.info(f"  Num examples = {len(val_dataset)}")

        with torch.no_grad():
            fixed_timesteps = [300, 500, 700]
            for i, timestep in enumerate(fixed_timesteps):
                for step, batch in enumerate(tqdm(val_dataloader)):
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["pixel_values"].to(device=unet.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.full((bsz,), timestep, device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    unmasked_x = batch["unmasked_x"].to(device=unet.device, dtype=weight_dtype)
                    unmasked_idx = batch["unmasked_idx"].to(device=unet.device, dtype=torch.long)
                    encoder_hidden_states = feature_encoder.encode(unmasked_x, unmasked_idx)

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(
                            prediction_type=args.prediction_type
                        )

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    masked_image_latents = vae.encode(
                        batch["masked_images"].to(device=unet.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_image_latents = masked_image_latents * vae.config.scaling_factor

                    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    mask = torch.nn.functional.interpolate(
                        batch["masks"],
                        size=(
                            args.resolution // vae_scale_factor,
                            args.resolution // vae_scale_factor,
                        ),
                    )
                    mask = mask.to(device=latents.device, dtype=weight_dtype)

                    latent_model_input = torch.cat(
                        [noisy_latents, mask, masked_image_latents], dim=1
                    )

                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        latent_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                    if args.snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        if noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)
                            
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                    epoch_val_loss_300_500_700[i] += loss.item()

        epoch_train_loss /= len(train_dataloader)
        for i in range(3):
            epoch_val_loss_300_500_700[i] /= len(val_dataloader)

        epoch_val_loss = sum(epoch_val_loss_300_500_700) / 3
        
        logger.info(f" **  Done epoch = {epoch} ** ")
        logger.info(f"  Training loss = {epoch_train_loss}")
        logger.info(f"  Validation loss = {epoch_val_loss}")

        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_loss_300": epoch_val_loss_300_500_700[0],
            "val_loss_500": epoch_val_loss_300_500_700[1],
            "val_loss_700": epoch_val_loss_300_500_700[2],
            "epoch": epoch,
        })

        if accelerator.is_main_process:
            if epoch > 100 and best_val_loss >= epoch_val_loss:
                best_val_loss = epoch_val_loss
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-bestVal-{epoch_val_loss}")
                accelerator.save_state(save_path)
                logger.info(f"Saved best validation state to {save_path}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        feature_encoder = unwrap_model(feature_encoder)

        pipeline = StableDiffusionInpaintPipelineWithOtherSideWithMET.from_pretrained(
            args.pretrained_model_name_or_path,
            feature_encoder=feature_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()
