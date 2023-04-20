"""
Script to generate results for our data
"""

import torch
import argparse
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

from results_generation_data import INPUT_IMAGE, INPUT_MASK, COMMAND, OUTPUT_IMAGE, INPUT_DIR, OUTPUT_DIR

parser = argparse.ArgumentParser(description='Run results for a given hyperparam set')
parser.add_argument('-t', '--text_guidance', type=float, required=True,
                    help='The strength of the text guidance (default=7.5)')
parser.add_argument('-f', '--mask_frequency', type=int, required=True,
                    help='The number of diffusion steps in between mask enforcement')
parser.add_argument('-m', '--mask_guidance', type=float, required=True,
                    help='Strength of mask guidance (between 0 and 1)')

if __name__ == '__main__':
    MODEL_ID = 'timbrooks/instruct-pix2pix'
    args = parser.parse_args()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID,
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    for i in range(len(INPUT_IMAGE)):
        print('Image: ', INPUT_IMAGE[i])
        print('Command: ', COMMAND[i])
        image = Image.open(INPUT_DIR + INPUT_IMAGE[i])
        mask_im = Image.open(INPUT_DIR + INPUT_MASK[i])
        mask_numpy = np.array(mask_im)
        mask_int = mask_numpy.astype(int)
        mask = mask_int / mask_int.max()
        images = pipe(COMMAND[i], image=image, mask=mask, mask_guidance_scale=args.mask_guidance,
                      guidance_scale=args.text_guidance,
                      mask_enforcement_frequency=args.mask_frequency).images
        result = images[0]
        result.save(OUTPUT_DIR+OUTPUT_IMAGE.format(args.text_guidance, args.mask_guidance, args.mask_frequency))
