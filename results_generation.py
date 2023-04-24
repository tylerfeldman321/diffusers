"""
Script to generate results for our data
"""

import argparse
import torch
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

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
    for i in range(len(COMMANDS)):
        image_path = INPUT_IMAGE[i // COMMANDS_PER_IMAGE]
        mask_path = INPUT_MASK[i // COMMANDS_PER_IMAGE] 
        command = COMMANDS[i]

        print(f'Image: {image_path}, Mask path: {mask_path}, command: {command}')
        image = Image.open(INPUT_DIR + image_path)
        if args.vanilla:
            images = pipe(command, image=image).images
            result = images[0]
            result.save(OUTPUT_DIR+VANILLA_OUTPUT[i])
        else:
            mask_im = Image.open(INPUT_DIR + mask_path).convert('RGB')
            mask_numpy = np.array(mask_im)
            mask_int = mask_numpy / mask_numpy.max()
            mask = mask_int.astype(int)
            images = pipe(command, image=image, mask=mask, mask_guidance_scale=args.mask_guidance,
                        guidance_scale=args.text_guidance,
                        mask_enforcement_frequency=args.mask_frequency).images
            result = images[0]
            result.save(OUTPUT_DIR+OUTPUT_IMAGE[i].format(args.text_guidance, args.mask_guidance, args.mask_frequency))
