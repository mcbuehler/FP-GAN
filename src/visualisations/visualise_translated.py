import matplotlib.pyplot as plt
from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, UnityDataLoader, RefinedUnityDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze
import os
import tensorflow as tf
from models.gazenet import GazeNetInference
from util.files import listdir
import cv2
from input.preprocessing import RefinedPreprocessor


class Visualisation:
    def __init__(self,
                 dl_original,
                 dl_refined,
                 sess,
                 path_ege,
                 n_images,
                 image_size,
                 norm,
                 normalise_gaze,
                 gazenet_name,
                 name_out='mpii_vs_refined-mpii.png',
                 do_draw_gaze=True):
        self.name_out = name_out
        self.dl_original = dl_original
        self.dl_refined = dl_refined
        self.path_ege = path_ege
        self.do_draw_gaze = do_draw_gaze
        self.gazenet_inference = GazeNetInference(
            sess,
            path_ege,
            n_images,
            image_size,
            norm,
            normalise_gaze,
            gazenet_name
        )
        self.color_refined = (255, 255, 0)
        self.color_true = (255, 0, 0)
        self.n_images = n_images

    def visualise(self, identifiers):
        raise NotImplementedError()

    def preprocess(self, images):
        preprocessor = RefinedPreprocessor(do_augmentation=False,
                                           eye_image_shape=(36, 60))
        images_preprocessed = [preprocessor.preprocess(img)[0] for img in
                               images]
        if len(images_preprocessed[0].shape) == 3:
            # We got a rgb image
            images_preprocessed = [np.mean(img, axis=2) for
                               img in images_preprocessed]

        images_preprocessed = [np.expand_dims(img, axis=2) for img in images_preprocessed]
        return images_preprocessed

    @staticmethod
    def dg(img, gaze, color, length=100, thickness=2):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=length, thickness=thickness, color=color,
        )

    def gray2rgb(self, image):
        image = np.expand_dims(image, axis=2)
        return np.repeat(image, 3, 2)


class M2UVisualisation(Visualisation):
    def sample_identifiers(self, path_original):
        samples = list()
        with h5py.File(path_original, 'r') as hf:
            for person_identifier in hf:
                n = hf[person_identifier]['image'].shape[0]
                index = np.random.randint(0, n)
                samples.append((person_identifier, index))
        return samples[:self.n_images]

    def get_data(self, identifiers=None):
        mpii_data = self.dl_original.get_data(identifiers)
        m2u_data = self.dl_refined.get_data(identifiers)
        return mpii_data, m2u_data

    def visualise(self, identifiers):
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(20, 20))

        # Axis labels
        cols = ["MPII (R)", "Refined MPII (S')"]
        for ax, col in zip(axes[0], cols):
            # Column title
            ax.set_title(col)
        # Labels for each image
        for ax, row in zip(axes[:, 0], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')
        for ax, row in zip(axes[:, 1], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')

        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)
        gaze_pred = self.gazenet_inference.predict_gaze(images_preprocessed)

        for i, (person_identifier, img_index) in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original = original_data[(person_identifier, img_index)]['eye']
            img_refined = refined_data[(person_identifier, img_index)]['eye']
            if self.do_draw_gaze:
                img_original = self.dg(img_original, original_data[(person_identifier, img_index)]['gaze'], color=self.color_true)
                img_refined = self.dg(img_refined, gaze_pred[i], color=self.color_true)
                img_refined = self.dg(img_refined, refined_data[(person_identifier, img_index)]['gaze'], color=self.color_refined)  #

            if len(img_refined.shape) == 2:
                img_refined = self.gray2rgb(img_refined)
            axes[row, 0].imshow(img_original)
            axes[row, 1].imshow(img_refined)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig(os.path.join('../visualisations/', self.name_out))


class U2MVisualisation(Visualisation):

    def sample_identifiers(self, path_original, path_refined):
        file_stems = listdir(path_original, postfix='.jpg',
                             return_postfix=False)
        index = list()
        c = 0
        while c < self.n_images:
            sample = np.random.choice(file_stems)
            if os.path.exists(os.path.join(path_refined, "{}.jpg".format(sample))):
                index.append(sample)
                c += 1
        return index

    def get_data(self, identifiers=None):
        return self.dl_original.get_data(identifiers), self.dl_refined.get_data(identifiers)

    def visualise(self, identifiers):
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=3, figsize=(20, 20))

        # Axis labels
        cols = ["Unity (S) full", "Unity (S) cropped", "Refined Unity (R')"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')
        for ax, row in zip(axes[:, 1], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')
        for ax, row in zip(axes[:, 2], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')

        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)
        gaze_pred = self.gazenet_inference.predict_gaze(images_preprocessed)

        for i, file_stem in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original_full = original_data[file_stem]['eye']
            img_original = refined_data[file_stem]['eye_original']
            img_refined = refined_data[file_stem]['eye']

            if len(img_refined.shape) == 2:
                img_refined = self.gray2rgb(img_refined)
                img_original = self.gray2rgb(img_original)

            if self.do_draw_gaze:
                img_original_full = self.dg(img_original_full, original_data[file_stem]['gaze'], length=400, thickness=5, color=self.color_true)
                img_original = self.dg(img_original, original_data[file_stem]['gaze'], color=self.color_true)
                img_refined = self.dg(img_refined, refined_data[file_stem]['gaze'], length=100, color=self.color_true)
                img_refined = self.dg(img_refined, gaze_pred[i], color=self.color_refined)

            axes[row, 0].imshow(img_original_full)
            axes[row, 1].imshow(img_original)
            axes[row, 2].imshow(img_refined)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig(os.path.join('../visualisations/', self.name_out))


if __name__ == "__main__":
    M2U = True
    M2U = False
    U2M = True
    # U2M = False

    # GAN identifier: {"U2M": (EGE Identifier, EGE name)}
    model_identifiers = {
        # "20181229-1345",
        # "20190105-1325",
        # "20190112-1740_ege_l5",
        "20190113-1455_ege_l8":
            {"U2M": ("20190114-2346_gazenet_u2m_bw_ege_l8", "gazenet_u2m_bw_ege_l8")}
        # "20190114-0959_ege_l15"
    }
    for model_identifier in model_identifiers:
        print("Processing model identifier {}...".format(model_identifier))
        if M2U:
            path_original = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
            path_refined = '../checkpoints/{}/refined_MPII2Unity'.format(model_identifier)
            path_ege = '../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw'
            dl_original = MPIIDataLoader(path_original)
            dl_refined = RefinedMPIIDataLoader(path_refined)

            with tf.Session() as sess:
                m2u_visualisation = M2UVisualisation(
                    dl_original=dl_original,
                    dl_refined=dl_refined,
                    path_ege=path_ege,
                    name_out='mpii_vs_refined_{}.png'.format(model_identifier),
                    n_images=15,
                    image_size=(36,60),
                    norm="batch",
                    normalise_gaze=False,
                    gazenet_name="gazenet_u_augmented_bw",
                    sess=sess
                )
                identifiers = m2u_visualisation.sample_identifiers(path_original)
                m2u_visualisation.visualise(identifiers)

        if U2M:
            path_original = '../data/UnityEyes'
            path_refined = '../checkpoints/{}/refined_Unity2MPII'.format(model_identifier)
            path_ege = '../checkpoints_gazenet/20190114-2346_gazenet_u2m_bw_ege_l8'
            dl_original = UnityDataLoader(path_original)
            dl_refined = RefinedUnityDataLoader(path_refined)

            with tf.Session() as sess:
                u2m_visualisation = U2MVisualisation(
                    dl_original=dl_original,
                    dl_refined=dl_refined,
                    path_ege=path_ege,
                    name_out='unity_vs_refined_{}.png'.format(model_identifier),
                    n_images=15,
                    image_size=(36,60),
                    norm="batch",
                    normalise_gaze=False,
                    gazenet_name="gazenet_u2m_bw_ege_l8",
                    sess=sess)
                identifiers = u2m_visualisation.sample_identifiers(
                    path_original, path_refined)
                u2m_visualisation.visualise(identifiers)
