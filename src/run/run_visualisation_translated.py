"""
Runs visualisations to compare original and translated images.
It is possible to draw eye gaze direction and landmarks on top of the images.
"""

import os
from collections import OrderedDict

import tensorflow as tf

from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, \
    UnityDataLoader, RefinedUnityDataLoader
from visualisations.visualise_translated import R2SVisualisation, \
    S2RVisualisation


# CONFIG VARIABLES
n_images = 7
do_plot_infobox = False
do_predict_eyegaze = True
do_draw_landmarks = True

R2S = True
S2R = True

path_elg = "../outputs_elg/ELG_i120x72_f60x36_n64_m3" \
           "/checkpoints/hourglass/model-1295134"


def run_r2s(model_identifier):
    path_original = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
    path_refined = '../checkpoints/{}/refined_MPII2Unity'.format(
        model_identifier)
    path_ege = '../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw'
    dl_original = MPIIDataLoader(path_original)
    dl_refined = RefinedMPIIDataLoader(path_refined)

    with tf.Session() as sess:
        r2s_visualisation = R2SVisualisation(
            dl_original=dl_original,
            dl_refined=dl_refined,
            path_ege=path_ege,
            path_elg=path_elg,
            name_out='mpii_vs_refined_{}.png'.format(model_identifier),
            n_images=n_images,
            image_size=(36, 60),
            norm="batch",
            normalise_gaze=False,
            gazenet_name="gazenet_u_augmented_bw",
            sess=sess,
            do_predict_gaze=do_predict_eyegaze,
            do_draw_landmarks=do_draw_landmarks,
            do_plot_infobox=do_plot_infobox
        )
        identifiers = r2s_visualisation.sample_identifiers(path_original)
        r2s_visualisation.visualise(identifiers)


def run_s2r(model_identifier):
    path_original = '../data/UnityEyes'
    path_refined = '../checkpoints/{}/refined_Unity2MPII'.format(
        model_identifier)
    if models[model_identifier]["S2R"]:
        ege_folder = models[model_identifier]["S2R"][0]
        gazenet_name = models[model_identifier]["S2R"][1]
        path_ege = os.path.join('../checkpoints_gazenet/', ege_folder)
    else:
        path_ege = None
        gazenet_name = None
    dl_original = UnityDataLoader(path_original)
    dl_refined = RefinedUnityDataLoader(path_refined)

    with tf.Session() as sess:
        s2r_visualisation = S2RVisualisation(
            dl_original=dl_original,
            dl_refined=dl_refined,
            path_ege=path_ege,
            path_elg=path_elg,
            name_out='unity_vs_refined_{}.png'.format(model_identifier),
            n_images=n_images,
            image_size=(36, 60),
            norm="batch",
            normalise_gaze=False,
            gazenet_name=gazenet_name,
            sess=sess,
            do_predict_gaze=do_predict_eyegaze,
            do_draw_landmarks=do_draw_landmarks,
            do_plot_infobox=do_plot_infobox)
        identifiers = s2r_visualisation.sample_identifiers(
            path_original, path_refined)
        s2r_visualisation.visualise(identifiers)


if __name__ == "__main__":
    # Here we list many of the trained GAN models that are available.
    # Comment out the ones you don't want to create visualisations for.


    # GAN identifier: {"S2R": (EGE Identifier, EGE name)}
    models = OrderedDict()
    # BASIC GAN
    # models["20181229-1345"] = {"S2R": ("20181230-1219_gazenet_s2r_augmented", "gazenet_s2r_augmented")}

    # SIMPLISTIC GAN
    # SELECTED SIMPLISTIC GAN
    # models["20190105-1325"] = {"S2R": ("20190108-1308_gazenet_s2r_augmented_bw", "gazenet_s2r_augmented_bw")}

    # EGE GAN
    # models["20190120-1420_ege_l1"] = {"S2R": ()}
    # models["20190112-1740_ege_l5"] = {"S2R": ()}
    # models["20190113-1455_ege_l8"] = {"S2R": ("20190114-2346_gazenet_s2r_bw_ege_l8", "gazenet_s2r_bw_ege_l8")}
    # models["20190114-0959_ege_l15"] = {"S2R": ()}
    # SELECTED EGE L30
    # models["20190118-1522_ege_l30"] = {"S2R": ("20190220-0859_gazenet_s2r_bw_ege_l30", "gazenet_s2r_bw_ege_l30")}
    # models["20190118-1542_ege_l50"] = {"S2R": ()}
    # models["20190120-1421_ege_l1_id1"] = {"S2R": ()}
    # models["20190115-1856_ege_l15_id10"] = {"S2R": ()}

    # # LM GANs
    # models["20190120-1423_lm_l1"] = {"S2R": ()}
    # models["20190116-2156_lm_l5"] = {"S2R": ()}
    # models["20190117-1430_lm_l8"] = {"S2R": ()}
    models["20190116-2305_lm_l15"] = {"S2R": ("20190220-1848_gazenet_s2r_bw_lm_l15", "gazenet_s2r_bw_lm_l15")}
    # models["20190120-1424_lm_l1_id1"] = {"S2R": ()}
    # models["20190117-1548_lm_l10_id5"] = {"S2R": ()}

    # models["20190122-2251_rhp_ege_30"] = {"S2R": ()}
    # models["20190123-0840_rhp_ege_20_id1"] = {"S2R": ()}
    # models["20190123-1455_rhp_id2"] = {"S2R": ()}

    for model_identifier in models.keys():
        print("Processing model identifier {}...".format(model_identifier))
        if R2S:
            run_r2s(model_identifier)

        if S2R:
            run_s2r(model_identifier)