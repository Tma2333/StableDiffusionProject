import typing as T

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from tqdm import tqdm


def visualize_diffusion_step_gif (image_list: T.List[np.ndarray] , prompt: str, file_name: str, dpi: int = 100):
    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.imshow(image_list[0])
    ax.axis('off')

    moviewriter = PillowWriter(fps=10)
    with moviewriter.saving(fig, file_name, dpi=dpi):
        for i in tqdm(range(len(image_list)), desc='rendeirng'):
            ax.set_title(f'prompt: {prompt}; step {i}')
            ax.imshow(image_list[i])
            moviewriter.grab_frame()