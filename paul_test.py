import torch
import numpy as np
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import time

from results_generation_data import COMMANDS_PER_IMAGE, INPUT_IMAGE, INPUT_MASK, COMMANDS, OUTPUT_IMAGE, INPUT_DIR, OUTPUT_DIR, VANILLA_OUTPUT

MODEL_ID = 'timbrooks/instruct-pix2pix'
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID,
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None)
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = Image.open(INPUT_DIR + 'statue.jpg')

images = pipe('Give them cleats', image=image, just_cycle=True).images
result = images[0]
result.save('PAUL_TEST.jpg')
print('Done')
