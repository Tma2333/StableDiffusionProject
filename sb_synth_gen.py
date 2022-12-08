import itertools
import json
import time
import uuid

import torch
import numpy as np
from tqdm import tqdm
import fire

from pipeline.text_to_image_pipeline import Text2ImagePipeline


#================================================
# Useful functions
#===============================================
def read_json(path: str) :
    with open(path, "r") as f:
        o = json.load(f)
    return o

def write_json(o, path) :
    with open(path, "w") as f:
        json.dump(o, f)


def doc_string_to_list (dstr):
    return [s.strip() for s in dstr.split(',')]


def gen_prompt_combination (template, *dstrs):
    all_variable_list = []
    for dstr in  dstrs:
        all_variable_list.append(doc_string_to_list(dstr))
    combined = list(itertools.product(*all_variable_list))
    prompts = []
    print(f'Generating {len(combined)} prompts')
    for var in combined:
        prompts.append(template.format(*var))
    return prompts


def cat_prompt_set_1(save_path):
    breed = """Abyssinian, Aegean, American Bobtail, American Curl, American Ringtail, American Shorthair, American Wirehair, Aphrodite Giant, Arabian Mau, Asian, Asian Semi-longhair, Australian Mist, Balinese, Bambino, Bengal, Birman, Bombay, Brazilian Shorthair, British Longhair, British Shorthair, Burmese, Burmilla, California Spangled, Chantilly-Tiffany, Chartreux, Chausie, Colorpoint Shorthair, Cornish Rex, Cymric, Manx Longhair, Long-haired Manx, Cyprus, Devon Rex, Donskoy, Don Sphynx, Dragon Li, Chinese Li Hua, Dwelf, Egyptian Mau, European Shorthair, Exotic Shorthair, Foldex, German Rex, Havana Brown, Highlander, Himalayan, Colorpoint Persian, Japanese Bobtail, Javanese, Colorpoint Longhair, Kanaani, Khao Manee, Kinkalow, Korat, Korean Bobtail, Korn Ja, Kurilian Bobtail, Kuril Islands Bobtail, Lambkin, LaPerm, Lykoi, Maine Coon, Manx, Mekong Bobtail, Minskin, Minuet, Munchkin, Nebelung, Norwegian Forest Cat, Ocicat, Ojos Azules, Oriental Bicolor, Oriental Longhair, Oriental Shorthair, Persian, Peterbald, Pixie-bob, Ragdoll, Raas, Russian Blue, Russian White, Russian Black, Russian Tabby, Sam Sawet, Savannah, Scottish Fold, Selkirk Rex, Serengeti, Serrade Petit, Siamese, Siberian Forest Cat, Neva Masquerade, Singapura, Snowshoe, Sokoke, Somali, Sphynx, Suphalak, Thai Lilac, Thai Blue Point, Thai Lilac Point, Tonkinese, Toybob, Toyger, Turkish Angora, Turkish Van, Turkish Vankedisi, Ukrainian Levkoy, York Chocolate"""


    furniture = """beds, cradle, trundle bed, cabinets, cellarette, court cupboard, cupboard, sideboard, vargueno, Barcelona chair, basket chair, bath chair, bench, Brewster chair, Carver chair, cathedra, chaise longue, cockfighting chair, confidante, couch, Cromwellian chair, curule chair, faldstool, farthingale chair, Gainsborough chair, inglenook, klismos, ladder-back chair, love seat, Morris chair, ottoman, pew, platform rocker, scissors chair, settee, settle, stool, taboret, throne, wainscot chair, Windsor chair, wing chair, chests, armoire, bureau, cassone, chest of drawers, coffer, commode, dresser, wardrobe, Act of Parliament clock, banjo clock, bracket clock, grandfather clock, ogee clock, pillar and scroll shelf clock, desks, bonheur du jour, carrel, davenport, lectern, prie-dieu, rolltop desk, secretary, other, bookcase, whatnot, tables, cabriole leg, candlestand, Carlton House table, console, dressing table, drop-leaf table, drum table, gateleg table, gueridon, highboy, lowboy, Parsons table, Pembroke table, tilt-top table, tripod, washstand"""

    prepositions = """at, on, in, at, on top of, behind, under, in front of, beside, next to, inside"""

    template = "{} cat {} the {}"

    all_prompts = gen_prompt_combination(template, breed, prepositions, furniture)
    print(f'Save path to {save_path}')
    write_json(all_prompts, save_path)



#================================================
# Generating funtions
#===============================================

def gen_images_from_prompts (batch_code, gpu_num, prmpt_path, begin, end, img_save_dir, meta_save_dir):
    all_prompts = read_json(prmpt_path)

    prompt_batch = all_prompts[int(begin): int(end)]

    torch_device = "cuda:{}".format(gpu_num)
    pipeline = Text2ImagePipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipeline.to(torch_device)

    batch_info = []
    for num, prompt in enumerate(prompt_batch):
        print(f'Generating {num+1}/{len(prompt_batch)}')

        height = 512                        # default height of Stable Diffusion
        width = 512                        # default width of Stable Diffusion

        num_inference_steps = 100            # Number of denoising steps

        guidance_scale = 7.5                # Scale for classifier-free guidance

        seed = int(time.time())
        generator = torch.Generator(device=torch_device)
        generator = generator.manual_seed(seed)

        batch_size = 1

        prompt = [prompt]

        out = pipeline(prompt=prompt, height=height, width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        output_type='pil')

        img_name = uuid.uuid4()


        batch_info.append({'name': f'{img_name}.png', 'prompt': prompt[0]})
        out[0].save(f'{img_save_dir}/{img_name}.png')
        write_json(batch_info, f'{meta_save_dir}/batch-{batch_code}.json')



if __name__ == "__main__":
    fire.Fire()
