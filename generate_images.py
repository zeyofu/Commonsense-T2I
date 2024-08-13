import os
from tqdm import tqdm
from PIL import Image
import torch
from datasets import load_dataset
import requests
import random

def load_dalle_model():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    return client

def load_lcm_model():
    from diffusers import DiffusionPipeline
    """Load a pretrained model"""
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", 
                                             custom_pipeline="latent_consistency_txt2img", 
                                             custom_revision="main", 
                                             revision="fb9c5d")
    pipe.to(torch_device="cuda", torch_dtype=torch.float32)
    return pipe

def load_sd3_model():
    from diffusers import StableDiffusion3Pipeline
    """Load a pretrained model"""
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def load_sd_xl_model():
    from diffusers import DiffusionPipeline
    """Load a pretrained model"""
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe = pipe.to("cuda")
    return pipe

def load_sd_21_model():
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    """Load a pretrained model"""
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

def load_flux_schnell_model():
    from diffusers import FluxPipeline
    """Load a pretrained model"""
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    return pipe

def load_flux_dev_model():
    from diffusers import FluxPipeline
    """Load a pretrained model"""
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    return pipe

def load_openjourneyv4_model():
    """Load a pretrained model"""
    from diffusers import StableDiffusionPipeline
    model_id = "prompthero/openjourney-v4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def load_playground25_model():
    """Load a pretrained model"""
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    return pipe

def generate_image_flux_schnell_model(pipe, prompt, neg_prompt=None):
    image = pipe(
        prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(random.randint(0, 1000000))
    ).images[0]
    return image

def generate_image_flux_dev_model(pipe, prompt, neg_prompt=None):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        output_type="pil",
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(random.randint(0, 1000000))
    ).images[0]
    return image

def generate_image_sd_model(pipe, prompt, neg_prompt=None):
    image = pipe(
        prompt=prompt, 
        use_negative_prompt=neg_prompt,
        num_inference_steps=28, 
        guidance_scale=7.0, 
        output_type="pil"
    ).images[0]
    return image

def generate_image_lcm_model(pipe, prompt, neg_prompt=None):
    image = pipe(
        prompt=prompt, 
        num_inference_steps=4, 
        guidance_scale=8.0, 
        output_type="pil"
    ).images[0]
    return image

def generate_image_dalle_no_revision_model(client, prompt, neg_prompt=None):
    no_revision_prompt_prefix = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
    # retry the current prompt if the connection to dalle-3 is broken up to three times
    for _ in range(3):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=no_revision_prompt_prefix+prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            img_url = response.data[0].url
            img_data = Image.open(requests.get(img_url, stream=True).raw)
            return img_data
        except Exception as e:
            print("Error: {}\n\nRegenerating the current prompt\n\n".format(e))
            continue


def generate_image_dalle_model(client, prompt, neg_prompt=None):
    # retry the current prompt if the connection to dalle-3 is broken up to three times
    for _ in range(3):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            img_url = response.data[0].url
            img_data = Image.open(requests.get(img_url, stream=True).raw)
            return img_data
        except Exception as e:
            print("Error: {}\n\nRegenerating the current prompt\n\n".format(e))
            continue

def generate_image_openjourneyv4_model(pipe, prompt, neg_prompt=None):
    image = pipe(prompt).images[0]
    return image

def generate_image_sd_21_model(pipe, prompt, neg_prompt=None):
    image = pipe(prompt).images[0]
    return image

def generate_image_playground25_model(pipe, prompt, neg_prompt=None):
    image = pipe(
        prompt=prompt, 
        num_inference_steps=50, 
        guidance_scale=3
    ).images[0]
    return image

def get_grid(imgs):
    """Given a list of generated images, generate a montage grid of them."""
    rows = 2
    cols = 2
    w, h = imgs[0].size
    grid_img = Image.new('RGB', size=(w*cols, h*rows))
    for i, img in enumerate(imgs):
        grid_img.paste(img, box=(i%cols*w, i//cols*h))
    return grid_img

def generate_images(pipe, model_generate_image, prompts, prompt_name, neg_prompts=None):
    """Iteratively generate images from prompts"""
    for i, prompt in tqdm(enumerate(prompts)):
        # skip if the image already exists
        grid_image_path = f"{output_image_dir}/{prompt_name}/{str(i + 1).zfill(4)}.jpg"
        # if os.path.exists(grid_image_path):
        #     continue
        
        print(f"The prompt to the {i}th image is: {prompt}")

        # generate images for four times for consistency
        images = []
        for j in range(4):
            if neg_prompts is None:
                image = model_generate_image(pipe, prompt)
            else:
                image = model_generate_image(pipe, prompt, neg_prompt=neg_prompts[i])
            images.append(image)
            image.save(f"{output_image_dir}/{prompt_name}/original/{str(i + 1).zfill(4)}-{j + 1}.jpg")

        # generate grid
        grid_img = get_grid(images)
        grid_img.save(grid_image_path)

if __name__ == "__main__":

    # Decide the model to load, choices are dalle, flux_schenel, flux_dev, sd_3, sd_xl, dalle_no_revision, LCMs, openjourneyv4, playground25, sd_21
    model_name = 'dalle'
    print(f"Generating images using {model_name} model")
    
    # Whether to use negative prompt for the model
    use_negative_prompt = False

    # Output directory for the generated images
    output_image_root = './generated_images'
    # output_image_root = '/shared/xingyu/projects/CommonsenseT2I/src/generated_images'
    
    # Create the output directory for the generated images and also the visualization grids
    output_image_dir = f'{output_image_root}/{model_name}{use_negative_prompt*"neg"}_images'
    os.makedirs(f"{output_image_dir}/prompt1_img/original", exist_ok=True)
    os.makedirs(f"{output_image_dir}/prompt2_img/original", exist_ok=True)

    # Load the model and the method to generate image
    load_model_methods = {'sd_3': load_sd3_model, 
                          'flux_schenel': load_flux_schnell_model, 
                          'flux_dev': load_flux_dev_model, 
                          'dalle3': load_dalle_model, 
                          'dalle3_no_revision': load_dalle_model, 
                          'LCMs': load_lcm_model, 
                          'sd_xl': load_sd_xl_model, 
                          'sd_21': load_sd_21_model,
                          'openjourneyv4': load_openjourneyv4_model, 
                          'playground25': load_playground25_model}
    model_generate_image_methods = {'sd_3': generate_image_sd_model, 
                                    'flux_schenel': generate_image_flux_schnell_model, 
                                    'flux_dev': generate_image_flux_dev_model, 
                                    'dalle3': generate_image_dalle_model, 
                                    'dalle3_no_revision': generate_image_dalle_no_revision_model, 
                                    'LCMs': generate_image_lcm_model, 
                                    'sd_xl': generate_image_sd_model, 
                                    'sd_21': generate_image_sd_21_model,
                                    'openjourneyv4': generate_image_openjourneyv4_model, 
                                    'playground25': generate_image_playground25_model}
    model_generate_image = model_generate_image_methods[model_name]
    pipe = load_model_methods[model_name]()

    # Load data and prompts
    data = load_dataset('CommonsenseT2I/CommonsensenT2I')['train']
    prompts1 = [d['prompt1'] for d in data]
    prompts2 = [d['prompt2'] for d in data]
    
    # Save the generated images and also grids for visualization
    if use_negative_prompt:
        generate_images(pipe, model_generate_image, prompts1, 'prompt1_img', prompts2)
        generate_images(pipe, model_generate_image, prompts2, 'prompt2_img', prompts1)
    else:
        generate_images(pipe, model_generate_image, prompts1, 'prompt1_img')
        generate_images(pipe, model_generate_image, prompts2, 'prompt2_img')