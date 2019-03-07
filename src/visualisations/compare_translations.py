"""
Visualisation scripts for comparing translations for different GANs.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from util.files import listdir


def gray2rgb(image):
    """
    Expands the dimensions of a gray-scale image such that it has three
        dimensions.
    Args:
        image: a single image

    Returns: image with three channels
    """
    image = np.expand_dims(image, axis=2)
    return np.repeat(image, 3, 2)


def create_path(gan_id, refined_folder):
    return os.path.join(
        "../checkpoints",
        gan_id,
        refined_folder
    )


def create_file_path(gan_id, refined_folder, image_stem, postfix):
    folder_path = create_path(gan_id, refined_folder)
    return os.path.join(folder_path,
        image_stem + postfix)


def sample_randomly(path, size=5):
    stems = listdir(path, postfix='_clean.jpg', return_postfix=False)
    return np.random.choice(stems, size=size)


def read_image(path):
    with open(path, 'rb') as image_file:
        image = plt.imread(image_file, format='jpg')
        image = gray2rgb(image)
    return image


def run_visualisation(gan_config, out_name):
    path_samples = create_path(*gan_config[0])
    identifiers_r2s = sample_randomly(path_samples)
    N_img = len(identifiers_r2s)
    N_gan = len(gan_config)
    figsize = 20
    fig, axes = plt.subplots(nrows=N_img, ncols=N_gan+1,
                             figsize=(figsize, figsize/2))
    plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

    # Axis labels
    cols = ["Original", "a)", "b)", "c)", "d)"]
    for ax, col in zip(axes[0], cols):
        # Column title
        ax.set_title(col, fontsize=30)

    for image_i in range(N_img):
        original_path = create_file_path(gan_config[0][0],
                                         gan_config[0][1],
                                         identifiers_r2s[image_i], "_clean.jpg")
        image = read_image(original_path)
        print(image.shape)
        axes[image_i, 0].imshow(image)
        axes[image_i, 0].axis('off')

        for gan_i in range(N_gan):
            refined_path = os.path.join("../checkpoints", gan_config[gan_i][0], gan_config[gan_i][1], identifiers_r2s[image_i] + ".jpg")
            image = read_image(refined_path)
            axes[image_i, gan_i+1].imshow(image)
            axes[image_i, gan_i+1].axis('off')

    plt.show()
    path_out = os.path.join('../visualisations/', out_name)
    print("Saving figure to {}...".format(path_out))
    plt.savefig(path_out, transparent=True)


gan_config_r2s = (
    ("20190105-1325", "refined_MPII2Unity"),
    ("20190118-1522_ege_l30", "refined_MPII2Unity"),
    ("20190116-2305_lm_l15", "refined_MPII2Unity"),
    ("20190115-1856_ege_l15_id10", "refined_MPII2Unity"),
)

gan_config_s2r = (
    ("20190105-1325", "refined_Unity2MPII"),
    ("20190118-1522_ege_l30", "refined_Unity2MPII"),
    ("20190116-2305_lm_l15", "refined_Unity2MPII"),
    ("20190115-1856_ege_l15_id10", "refined_Unity2MPII"),
)
run_visualisation(gan_config_r2s, out_name="compare_translations_r2s.png")
run_visualisation(gan_config_s2r, out_name="compare_translations_s2r.png")

