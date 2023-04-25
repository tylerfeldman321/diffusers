"""
Script to generate results for our data
"""

import argparse
import torch
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import time

from results_generation_data import COMMANDS_PER_IMAGE, INPUT_IMAGE, INPUT_MASK, COMMANDS, OUTPUT_IMAGE, INPUT_DIR, OUTPUT_DIR, VANILLA_OUTPUT

parser = argparse.ArgumentParser(description='Run results for a given hyperparam set')
parser.add_argument("--vanilla", action="store_true",
                    help="Will produce just the normal (unmasked) IP2P image")
parser.add_argument('-t', '--text_guidance', type=float,
                    help='The strength of the text guidance (default=7.5)')
parser.add_argument('-f', '--mask_frequency', type=int,
                    help='The number of diffusion steps in between mask enforcement')
parser.add_argument('-m', '--mask_guidance', type=float,
                    help='Strength of mask guidance (between 0 and 1)')

if __name__ == '__main__':
    MODEL_ID = 'timbrooks/instruct-pix2pix'
    args = parser.parse_args()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID,
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    run_times = []

    for i in range(len(COMMANDS)):
        image_path = INPUT_IMAGE[i // COMMANDS_PER_IMAGE]
        mask_path = INPUT_MASK[i // COMMANDS_PER_IMAGE] 
        command = COMMANDS[i]

        if args.vanilla:
            output_dir_vanilla = os.path.join(OUTPUT_DIR, 'vanilla')
            if not os.path.exists(output_dir_vanilla):
                os.makedirs(output_dir_vanilla)
            output_path = os.path.join(output_dir_vanilla, VANILLA_OUTPUT[i])
        else:
            output_dir_mask_enforced = os.path.join(OUTPUT_DIR, '{}_{}_{}'.format(args.text_guidance, args.mask_guidance, args.mask_frequency))
            if not os.path.exists(output_dir_mask_enforced):
                os.makedirs(output_dir_mask_enforced)
            os.path.join(output_dir_mask_enforced, OUTPUT_IMAGE[i].format(args.text_guidance, args.mask_guidance, args.mask_frequency))

        print(f'Image: {image_path}, Mask path: {mask_path}, command: {command}, output path: {output_path}')
        image = Image.open(INPUT_DIR + image_path)

        if args.vanilla:
            start_time = time.time()
            images = pipe(command, image=image).images
            end_time = time.time()
            result = images[0]
            result.save(output_path)
        else:
            mask_im = Image.open(INPUT_DIR + mask_path).convert('RGB')
            mask_numpy = np.array(mask_im)
            mask_int = mask_numpy / mask_numpy.max()
            mask = mask_int.astype(int)
            start_time = time.time()
            images = pipe(command, image=image, mask=mask, mask_guidance_scale=args.mask_guidance,
                        guidance_scale=args.text_guidance,
                        mask_enforcement_frequency=args.mask_frequency).images
            end_time = time.time()
            result = images[0]
            result.save(output_path)

        run_time = end_time - start_time
        run_times.append(run_time)

    avg_runtime = sum(run_times) / len(run_times)
    print(f'Average runtime: {avg_runtime}, it/s: {100 / avg_runtime}')
