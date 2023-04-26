from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import numpy as np

def run(image_path, mask_path, mask_enforcement_period, mask_guidance_scale, text_guidance_scale, prompt, returned_image):
    MODEL_ID = 'timbrooks/instruct-pix2pix'
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(MODEL_ID,
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None)
    pipe.to('cuda')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    image_path = image_path
    mask_path = mask_path 
    command = prompt
    output_path = returned_image

    print(f'Image: {image_path}, Mask path: {mask_path}, command: {command}, output path: {output_path}')
    image = Image.open(image_path)
    mask_im = Image.open(mask_path).convert('RGB')
    mask_numpy = np.array(mask_im)
    mask_int = mask_numpy / mask_numpy.max()
    mask = mask_int.astype(int)
    images = pipe(command, image=image, mask=mask, mask_guidance_scale=mask_guidance_scale,
                    guidance_scale=text_guidance_scale,
                    mask_enforcement_frequency=mask_enforcement_period).images

    result = images[0]
    result.save(output_path)

if __name__=="__main__":
    image_path = "/Users/easoplee/Desktop/diffusers/inputs/statue.jpg"
    mask_path = "/Users/easoplee/Desktop/diffusers/inputs/statue_mask.jpg"
    mask_enforcement_period = 10
    mask_guidance_scale = 0.2
    text_guidance_scale = 5.0
    prompt = 'Give him black hair'
    returned_image = "statue_black_hair_output.jpg"

    run(image_path, mask_path, mask_enforcement_period, mask_guidance_scale, text_guidance_scale, prompt, returned_image)