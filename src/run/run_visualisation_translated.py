import os
from collections import OrderedDict

import tensorflow as tf

from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, \
    UnityDataLoader, RefinedUnityDataLoader
from visualisations.visualise_translated import M2UVisualisation, \
    U2MVisualisation


# CONFIG VARIABLES
n_images = 7
do_plot_infobox = False
do_predict_eyegaze = True
do_draw_landmarks = True

M2U = True
U2M = True

path_elg = "../outputs_elg/ELG_i120x72_f60x36_n64_m3" \
           "/checkpoints/hourglass/model-1295134"


def run_m2u(model_identifier):
    path_original = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
    path_refined = '../checkpoints/{}/refined_MPII2Unity'.format(
        model_identifier)
    path_ege = '../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw'
    dl_original = MPIIDataLoader(path_original)
    dl_refined = RefinedMPIIDataLoader(path_refined)

    with tf.Session() as sess:
        m2u_visualisation = M2UVisualisation(
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
        identifiers = m2u_visualisation.sample_identifiers(path_original)
        m2u_visualisation.visualise(identifiers)


def run_u2m(model_identifier):
    path_original = '../data/UnityEyes'
    path_refined = '../checkpoints/{}/refined_Unity2MPII'.format(
        model_identifier)
    if models[model_identifier]["U2M"]:
        ege_folder = models[model_identifier]["U2M"][0]
        gazenet_name = models[model_identifier]["U2M"][1]
        path_ege = os.path.join('../checkpoints_gazenet/', ege_folder)
    else:
        path_ege = None
        gazenet_name = None
    dl_original = UnityDataLoader(path_original)
    dl_refined = RefinedUnityDataLoader(path_refined)

    with tf.Session() as sess:
        u2m_visualisation = U2MVisualisation(
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
        identifiers = u2m_visualisation.sample_identifiers(
            path_original, path_refined)
        u2m_visualisation.visualise(identifiers)


if __name__ == "__main__":
    # GAN identifier: {"U2M": (EGE Identifier, EGE name)}
    models = OrderedDict()
    # BASIC GAN
    # models["20181229-1345"] = {"U2M": ("20181230-1219_gazenet_u2m_augmented", "gazenet_u2m_augmented")}
    # SIMPLISTIC GAN
    # SELECTED SIMPLISTIC GAN
    # models["20190105-1325"] = {"U2M": ("20190108-1308_gazenet_u2m_augmented_bw", "gazenet_u2m_augmented_bw")}
    # EGE GAN
    # models["20190120-1420_ege_l1"] = {"U2M": ()}
    # models["20190112-1740_ege_l5"] = {"U2M": ()}
    # models["20190113-1455_ege_l8"] = {"U2M": ("20190114-2346_gazenet_u2m_bw_ege_l8", "gazenet_u2m_bw_ege_l8")}
    # models["20190114-0959_ege_l15"] = {"U2M": ()}
    # SELECTED EGE L30
    # models["20190118-1522_ege_l30"] = {"U2M": ("20190220-0859_gazenet_u2m_bw_ege_l30", "gazenet_u2m_bw_ege_l30")}
    # models["20190118-1542_ege_l50"] = {"U2M": ()}
    #
    # models["20190120-1421_ege_l1_id1"] = {"U2M": ()}
    # models["20190115-1856_ege_l15_id10"] = {"U2M": ()}
    # # LM GANs
    # models["20190120-1423_lm_l1"] = {"U2M": ()}
    # models["20190116-2156_lm_l5"] = {"U2M": ()}
    # models["20190117-1430_lm_l8"] = {"U2M": ()}
    models["20190116-2305_lm_l15"] = {"U2M": ("20190220-1848_gazenet_u2m_bw_lm_l15", "gazenet_u2m_bw_lm_l15")}
    # #
    # # TODO models["20190120-1424_lm_l1_id1"] = {"U2M": ()}
    # models["20190117-1548_lm_l10_id5"] = {"U2M": ()}

    # models["20190122-2251_rhp_ege_30"] = {"U2M": ()}
    # models["20190123-0840_rhp_ege_20_id1"] = {"U2M": ()}
    # models["20190123-1455_rhp_id2"] = {"U2M": ()}

    for model_identifier in models.keys():
        print("Processing model identifier {}...".format(model_identifier))
        if M2U:
            run_m2u(model_identifier)

        if U2M:
            run_u2m(model_identifier)