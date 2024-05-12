import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import numpy as np
from PIL import Image

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16, use_safetensors=True
)
# pipeline = AutoPipelineForImage2Image.from_pretrained(
    # "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

path = "/DATA1/zhuyunwei/KITTI-360/chunks/2013_05_28_drive_0000_sync/21/rgbs/0000006588.png"
init_image = load_image(path)
H,W = init_image.height,init_image.width
init_image = init_image.resize((512,512))
from PIL import ImageFilter
radius = 2
noisy_image = init_image.filter(ImageFilter.GaussianBlur(radius))

# mean = 0
# stddev = 10  # Adjust the standard deviation to control the amount of noise
# image_array = np.array(init_image)
# noise = np.random.normal(mean, stddev, image_array.shape)
# # print(noise)
# noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
# noisy_image = Image.fromarray(noisy_image_array)

prompt = "street view photo, cars, road, buildings, plants, epic"
image = pipeline(prompt, image=noisy_image,strength=0.2).images[0]
grid = make_image_grid([init_image.resize((W,H)),noisy_image.resize((W,H)),image.resize((W,H))], rows=3,cols=1)
grid.save("./vis/grid.jpg")