from diffusers import DiffusionPipeline


pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",  cache_dir="./custom_cache")

prompt = "An astronaut riding a green horse"

images = pipeline(prompt=prompt).images[0]