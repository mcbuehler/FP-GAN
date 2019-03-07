from models.gazenet import GazeNetInference
import tensorflow as tf
from visualisations.data_loading import UnityDataLoader, RefinedUnityDataLoader
from visualisations.visualise_translated import Visualisation
import matplotlib.pyplot as plt
import os
import numpy as np


COLOR_GAZE = (255, 0, 0)
COLOR_LANDMARKS = (0, 255, 255)


def save_img(img, path, filename):
    plt.axis('off')
    plt.imshow(img)
    path_out = os.path.join(path, filename)
    plt.savefig(path_out, bbox_inches='tight')


def run_ege(model_identifier, image_id, out_path):
    path_refined = '../checkpoints/{}/refined_Unity2MPII'.format(
        model_identifier)

    dl_refined = RefinedUnityDataLoader(path_refined)
    data = dl_refined.get_data([image_id])[image_id]

    img_original_out = data['eye_original']
    img_original_out = Visualisation.gray2rgb(img_original_out)

    img_refined_out = data['eye']
    img_refined_out = Visualisation.gray2rgb(img_refined_out)

    save_img(img_refined_out, out_path, "eye_59_u2m_refined.png")

    img_refined_out = Visualisation.dg(img_refined_out, data['gaze'], COLOR_GAZE)
    save_img(img_refined_out, out_path, "eye_59_u2m_refined_gaze.png")

    save_img(img_original_out, out_path, "eye_59_u2m_original.png")

    img_original_out = Visualisation.dg(img_original_out, data['gaze'], COLOR_GAZE)
    save_img(img_original_out, out_path, "eye_59_u2m_original_gaze.png")


def run_lm(model_identifier, image_id, out_path):
    path_refined = '../checkpoints/{}/refined_Unity2MPII'.format(
        model_identifier)

    landmarks_true = [[11.812553, 22.077623],
       [13.84647 , 15.511036],
       [29.53195 , 10.007287],
       [46.521633, 15.425174],
       [50.398117, 20.804216],
       [45.190575, 25.201675],
       [31.166977, 27.84388 ],
       [17.72636 , 25.53534 ],
       [29.91222 , 18.223734],
       [32.322052, 11.001873],
       [38.472996,  7.858983],
       [44.761898, 10.636148],
       [47.504795, 17.706514],
       [45.094933, 24.928375],
       [38.94399 , 28.071266],
       [32.655117, 25.29413 ],
       [39.835182, 17.960094],
       [28.685736, 18.007645]]

    landmarks_refined = [[x + np.random.normal(0, 1), y + np.random.normal(0, 1)] for x, y in landmarks_true]

    dl_refined = RefinedUnityDataLoader(path_refined)
    data = dl_refined.get_data([image_id])[image_id]

    img_original_out = data['eye_original']
    img_original_out = Visualisation.gray2rgb(img_original_out)

    img_refined_out = data['eye']
    img_refined_out = Visualisation.gray2rgb(img_refined_out)

    img_refined_out = Visualisation.draw_landmarks(img_refined_out, landmarks_true, COLOR_LANDMARKS)
    save_img(img_refined_out, out_path, "eye_59_u2m_refined_lm.png")

    img_original_out = Visualisation.draw_landmarks(img_original_out, landmarks_refined, COLOR_LANDMARKS)
    save_img(img_original_out, out_path, "eye_59_u2m_original_lm.png")


if __name__ == "__main__":
    out_path = "/home/marcello/studies/2018-3_herbstsemester/research-in-datascience/wookie/report/"
    model_identifier = "20190118-1522_ege_l30"
    image_id = "59"
    # run_ege(model_identifier, image_id, out_path)
    run_lm(model_identifier, image_id, out_path)
