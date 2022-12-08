import json
from pathlib import Path
import shutil

from pipeline.text_to_image_pipeline import Text2ImagePipeline
import torch
from diffusers import LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler

lmsd_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

torch_device = "cuda:9" if torch.cuda.is_available() else "cpu"
pipeline = Text2ImagePipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to(torch_device)

pipeline.scheduler = lmsd_scheduler


source = {
    'match_sticks': '/deep2/u/yma42/StableDiffusionProject/docs/match_sticks_seed5/meta.json',
    # 'the_steadfast_tin_soldier': '/deep2/u/yma42/StableDiffusionProject/docs/the_steadfast_tin_soldier_seed5/meta.json',
    # 'jack_and_the_beanstalk': '/deep2/u/yma42/StableDiffusionProject/docs/jack_and_the_beanstalk_seed5/meta.json',
    # 'the_ugly_duckling': '/deep2/u/yma42/StableDiffusionProject/docs/the_ugly_duckling_seed5/meta.json'
    # 'midas': '/deep2/u/yma42/StableDiffusionProject/docs/midas_seed5/meta.json',
    # 'matchgirl':'/deep2/u/yma42/StableDiffusionProject/docs/matchgirl_seed5/meta.json',
    # 'ant_grasshopper':'/deep2/u/yma42/StableDiffusionProject/docs/ant_grasshopper_seed5/meta.json',
    # 'sleeping_beauty': '/deep2/u/yma42/StableDiffusionProject/docs/sleeping_beauty_seed5/meta.json'
    # 'narcissus': '/deep2/u/yma42/StableDiffusionProject/docs/narcissus_seed5/meta.json',
    # 'icarus': '/deep2/u/yma42/StableDiffusionProject/docs/icarus_seed5/meta.json',
    # 'red_riding_hood': '/deep2/u/yma42/StableDiffusionProject/docs/red_riding_hood_seed5/meta.json'
}

seeds = [5, 132, 192, 134, 42, 19, 23, 99, 10, 223,49102, 3103, 4952, 402, 420, 69]

for name_base, meta_source in source.items():
    for seed in seeds:
        guidance_scale = 12
        guidance_tag = '' if guidance_scale == 7.5 else '_high_guidance'

        root_dir = Path('/deep2/u/yma42/StableDiffusionProject/docs/run_12_03_match_sticks') / f'{name_base}_seed{seed}'
        root_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src=meta_source, dst=str(root_dir/'meta.json'))

        json_path = root_dir/'meta.json'

        with open(str(json_path), "r") as f:
            meta = json.load(f)
                
        for i, json_dict in enumerate(meta):
            prompt = [json_dict['summary']]

            height = 512                        # default height of Stable Diffusion
            width = 512                        # default width of Stable Diffusion

            num_inference_steps = 100            # Number of denoising steps
            
            generator = torch.Generator(device=torch_device)
            generator = generator.manual_seed(seed)

            batch_size = 1

            out = pipeline(prompt=prompt, height=height, width=width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            output_type='pil')

            
            img_name = f'{name_base}_{i}'
            meta[i]['image'] = f'{img_name}.png'
            out[0].save(str(json_path.parent/f'{img_name}{guidance_tag}.png'))

        with open(json_path, "w") as f:
            json.dump(meta, f)
        
