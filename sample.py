from diffusers.pipelines import StableDiffusionPipeline
import torch

sample_nums = 1000
batch_size = 10
prompt = "fanart of a bullywug"
save_dir = "data/joetest/class"


if __name__ == "__main__":
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    model = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, cache_dir="./.cache").to(device)

    datasets = [prompt] * sample_nums
    datasets = [datasets[x:x+batch_size] for x in range(0, sample_nums, batch_size)]
    id = 0

    for text in datasets:
        with torch.no_grad():
            with torch.autocast("cuda"):
                generator = torch.Generator("cuda").manual_seed(id)
                images = model(text, height=512, width=512, num_inference_steps=50, generator=generator)["sample"]

        for image in images:
            image.save(f"{save_dir}/{id}.png")
            id += 1
