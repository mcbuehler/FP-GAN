from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, UnityDataLoader, RefinedUnityDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze, draw_gaze_py
import os
import tensorflow as tf
from models.gazenet import GazeNetInference
from util.files import listdir
from util.gaze import angular_error, euclidean_error, mse
from input.preprocessing import RefinedPreprocessor
from models.elg import ELGInference


class Visualisation:
    """
    Base class for visualising translated images.
    """
    def __init__(self,
                 dl_original,
                 dl_refined,
                 sess,
                 path_ege,
                 path_elg,
                 n_images,
                 image_size,
                 norm,
                 normalise_gaze,
                 gazenet_name,
                 name_out='mpii_vs_refined-mpii.png',
                 do_draw_gaze=True):
        """
        Args:
            dl_original: DataLoader for original images
            dl_refined: DataLoader for refined images
            sess: tensorflow session
            path_ege: path to model for eye gaze estimation
            path_elg: path to model for landmark detection
            n_images: number of images to display
            image_size: image size to load
            norm: TODO probably not used
            normalise_gaze: whether to normalise the gaze or not
            gazenet_name: corresponds to scope of saved weigths
            name_out: filename of saved figure
            do_draw_gaze: indicates whether we draw the gaze direction or not
        """
        self.name_out = name_out
        self.dl_original = dl_original
        self.dl_refined = dl_refined
        self.color_predicted = (255, 255, 0)
        self.color_true = (255, 0, 0)

        self.color_lm_refined = (0, 255, 255)

        self.n_images = n_images
        self.do_draw_gaze = do_draw_gaze
        self.predict_gaze = False
        self.predict_lm = False

        if path_ege and gazenet_name:
            self.predict_gaze = True
            self.path_ege = path_ege
            self.gazenet_inference = GazeNetInference(
                sess,
                path_ege,
                n_images,
                image_size,
                norm,
                normalise_gaze,
                gazenet_name
            )
        if path_elg:
            self.predict_lm = True
            self.elg_inference = ELGInference(
                sess,
                path_elg,
                batch_size=n_images,
                image_size=image_size
            )

    def visualise(self, identifiers):
        raise NotImplementedError()

    def preprocess(self, images):
        """
        We apply simple preprocessing in order for consistency.
        Args:
            images: tensor containing images.
                images may be both gray-scale or RGB.

        Returns: preprocessed images

        """
        preprocessor = RefinedPreprocessor(do_augmentation=False,
                                           eye_image_shape=(36, 60))
        images_preprocessed = [preprocessor.preprocess(img)[0] for img in
                               images]
        if len(images_preprocessed[0].shape) == 3:
            # We got a rgb image, so we convert it to gray-scale
            images_preprocessed = [self.rgb2gray(img) for
                               img in images_preprocessed]
        # We want to return a gray-scale image that uses all three RGB channels
        images_preprocessed = [self.gray2rgb(img) for img in images_preprocessed]
        return images_preprocessed

    @staticmethod
    def dg(img, gaze, color, length=100, thickness=2):
        """
        Draws a single eye gaze direction vector in img.
        We calculate a single vector from pitch and yaw.
        Args:
            img: a single image
            gaze: 2D pitch / yaw gaze direction
            color: color of gaze arrow
            length: length of gaze arrow
            thickness: thickness of gaze arrow

        Returns: img with gaze direction
        """
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=length, thickness=thickness, color=color,
        )

    @staticmethod
    def dg_py(img, gaze, color, length=100, thickness=2):
        """
        Draws two eye gaze vectors in img (pitch and yaw).
        Args:
            img: a single image
            gaze: 2D pitch / yaw gaze direction
            color: color of gaze arrow
            length: length of gaze arrow
            thickness: thickness of gaze arrow

        Returns: img with gaze direction vectors
        """
        return draw_gaze_py(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=length, thickness=thickness, color=color,
        )

    @staticmethod
    def draw_landmarks(img, coordinates, color):
        """
        Visualises the landmarks on img
        Args:
            img: an image
            coordinates: output of landmark detection model
            color: color of landmarks

        Returns:

        """
        import cv2
        for c in coordinates:
            # type 2: stars
            # size: in pixels
            img = cv2.drawMarker(img, tuple(c), color=color, markerType=2, markerSize=1)
        return img

    def gray2rgb(self, image):
        """
        Expands the dimensions of a gray-scale image such that it has three
            dimensions.
        Args:
            image: a single image

        Returns: image with three channels
        """
        image = np.expand_dims(image, axis=2)
        return np.repeat(image, 3, 2)

    def rgb2gray(self, image):
        """
        Converts an RGB image to gray-scale
        Args:
            image: a single image

        Returns: gray-scale image (single channel)
        """
        image = np.mean(image, axis=2)
        return self.gray2rgb(image)

    @staticmethod
    def format_gaze(gaze):
        """
        Helper function to format eye gaze direction
        Args:
            gaze: pitch / yaw

        Returns: formatted string
        """
        return "{:.2f}, {:.2f}".format(gaze[0], gaze[1])

    def _gaze_error(self, images_preprocessed, gaze_true):
        """
        Calculates the gaze error
        Args:
            images_preprocessed: images ready for eye gaze prediction
            gaze_true: true labels

        Returns: predicted gaze,
            angular error, euclidean error, mean squared error
        """
        gaze_pred = self.gazenet_inference.predict_gaze(images_preprocessed)
        gaze_error = angular_error(gaze_pred, gaze_true)
        eucl_gaze_error = euclidean_error(gaze_pred, gaze_true)
        ms_error = mse(gaze_pred, gaze_true)
        return gaze_pred, gaze_error, eucl_gaze_error, ms_error

    def _predict_landmarks(self, images_preprocessed):
        """
        Predicts the landmarks for preprocessed images
        Args:
            images_preprocessed: images ready for landmark prediction

        Returns:
        """
        return self.elg_inference.predict(images_preprocessed)


class M2UVisualisation(Visualisation):
    """
    Visualisation class for real images and their translations.
    """

    def sample_identifiers(self, path_original):
        """
        Create random samples from h5 file.
        Args:
            path_original: path to original h5 file

        Returns: random samples of identifiers as list of tuples
        """
        samples = list()
        with h5py.File(path_original, 'r') as hf:
            for person_identifier in hf:
                n = hf[person_identifier]['image'].shape[0]
                index = np.random.randint(0, n)
                samples.append((person_identifier, index))
        return samples[:self.n_images]

    def get_data(self, identifiers=None):
        """
        Load data using DataLoaders
        Args:
            identifiers: tuples of identifiers
        Returns: data for original images, data for refined images
        """
        mpii_data = self.dl_original.get_data(identifiers)
        m2u_data = self.dl_refined.get_data(identifiers)
        return mpii_data, m2u_data

    def visualise(self, identifiers, show=False, save=True):
        """
        Create the visualisations for given identifiers
        Args:
            identifiers:  tuples of identifiers
            show: whether to show result
            save: whether to save figure

        Returns:
        """
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=3, figsize=(20, 20))

        # Axis labels
        cols = ["MPII (R)", "Refined MPII (S')"]
        for ax, col in zip(axes[0], cols):
            # Column title
            ax.set_title(col)

        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)

        if self.predict_gaze:
            gaze_true = np.array([original_data[
                                      (pi, ii)][
                                      'gaze'] for pi, ii in identifiers])
            gaze_pred, gaze_error, eucl_gaze_error, ms_error = self._gaze_error(
                images_preprocessed,
                gaze_true
            )

        if self.predict_lm:
            landmarks_pred_refined = self._predict_landmarks(images_preprocessed)

        # Sort by ascending prediction error (if applicable)
        if self.predict_gaze:
            indices = np.argsort(gaze_error)
        else:
            indices = np.argsort(identifiers)

        for row, i in enumerate(indices):
            person_identifier = identifiers[i][0]
            img_index = identifiers[i][1]

            info_txt = ""
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original = original_data[(person_identifier, img_index)]['eye']
            img_refined = refined_data[(person_identifier, img_index)]['eye']

            if len(img_refined.shape) == 2:
                img_original = self.rgb2gray(img_original)
                img_refined = self.gray2rgb(img_refined)

            if self.predict_lm:
                self.draw_landmarks(img_refined, landmarks_pred_refined[i],
                                    self.color_lm_refined)

            if self.do_draw_gaze:
                img_original = self.dg(img_original, original_data[(person_identifier, img_index)]['gaze'], color=self.color_true)
                img_refined = self.dg_py(img_refined, refined_data[(person_identifier, img_index)]['gaze'], color=self.color_true)
                info_txt = "{} \n true pitch / yaw: {}".format(info_txt, self.format_gaze(original_data[(person_identifier, img_index)]['gaze']))
                if self.predict_gaze:
                    # img_refined = self.dg(img_refined, gaze_pred[i], color=self.color_predicted)
                    img_refined = self.dg_py(img_refined, gaze_pred[i], color=self.color_predicted)
                    info_txt = "{} \n predicted pitch / yaw: {} " \
                               "\n error angular / euclidean / mse: {:.2f} / {:.2f} / {:.2f}".\
                        format(info_txt, self.format_gaze(gaze_pred[i]), gaze_error[i], eucl_gaze_error[i], ms_error[i])


            axes[row, 0].imshow(img_original)
            axes[row, 1].imshow(img_refined)
            TextBox(axes[row, 2], person_identifier, initial=info_txt)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        if show:
            plt.show()
        if save:
            plt.savefig(os.path.join('../visualisations/', self.name_out))
        plt.close(fig)


class U2MVisualisation(Visualisation):
    """
    Visualisation class for synthetic images and their translations.
    """

    def sample_identifiers(self, path_original, path_refined):
        """
        Create random samples for dataset.
        Args:
            path_original: path to folder containing images
            path_refined: path to folder containing translated images

        Returns: random samples of identifiers (size: self.n_images)
        """
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
        """
        Loads the data from DataLoaders
        Args:
            identifiers: list of ids

        Returns: data for original, data for refined
        """
        return self.dl_original.get_data(identifiers), self.dl_refined.get_data(identifiers)

    def visualise(self, identifiers, show=False, save=True):
        """
        Create the visualisations for given identifiers
        Args:
            identifiers:  ids
            show: whether to show result
            save: whether to save figure

        Returns:
        """
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=4, figsize=(20, 20))

        # Axis labels
        cols = ["Unity (S) full", "Unity (S) cropped", "Refined Unity (R')"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)
        if self.predict_gaze:
            gaze_true = np.array([original_data[i][
                                           'gaze'] for i in identifiers])
            gaze_pred, gaze_error, eucl_gaze_error, ms_error = self._gaze_error(
                images_preprocessed,
                gaze_true
            )

        if self.predict_lm:
            landmarks_pred_refined = self._predict_landmarks(images_preprocessed)


        # Sort by ascending prediction error (if applicable)
        if self.predict_gaze:
            indices = np.argsort(gaze_error)
        else:
            indices = np.argsort(identifiers)

        for row, i in enumerate(indices):
            info_txt = ""
            file_stem = identifiers[i]
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original_full = original_data[file_stem]['eye']
            img_original = refined_data[file_stem]['eye_original']
            img_refined = refined_data[file_stem]['eye']

            if len(img_refined.shape) == 2:
                img_refined = self.gray2rgb(img_refined)
                img_original = self.gray2rgb(img_original)

            if self.predict_lm:
                self.draw_landmarks(img_refined, landmarks_pred_refined[i],
                                    self.color_lm_refined)

            if self.do_draw_gaze:
                img_original_full = self.dg(img_original_full, original_data[file_stem]['gaze'], length=400, thickness=5, color=self.color_true)
                img_original = self.dg(img_original, original_data[file_stem]['gaze'], color=self.color_true)
                img_refined = self.dg_py(img_refined, refined_data[file_stem]['gaze'], color=self.color_true)
                info_txt = "{} \n true pitch / yaw: {}".format(info_txt,
                                                        self.format_gaze(
                                                            original_data[file_stem][
                                                                'gaze']))
                if self.predict_gaze:
                    img_refined = self.dg_py(img_refined, gaze_pred[i], color=self.color_predicted)
                    info_txt = "{} \n predicted pitch / yaw: {} " \
                               "\n error angular / euclidean / mse: {:.2f} / {:.2f} / {:.2f}".\
                        format(info_txt, self.format_gaze(gaze_pred[i]), gaze_error[i], eucl_gaze_error[i], ms_error[i])

            axes[row, 0].imshow(img_original_full)
            axes[row, 1].imshow(img_original)
            axes[row, 2].imshow(img_refined)
            TextBox(axes[row, 3], file_stem, initial=info_txt)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)
        if show:
            plt.show()
        if save:
            plt.savefig(os.path.join('../visualisations/', self.name_out))
        plt.close(fig)


if __name__ == "__main__":
    M2U = True
    # M2U = False
    U2M = True
    # U2M = False

    path_elg = "../outputs_elg/ELG_i120x72_f60x36_n64_m3/checkpoints/hourglass/model-1295134"

    # GAN identifier: {"U2M": (EGE Identifier, EGE name)}
    models = OrderedDict()
    # BASIC GAN
    # models["20181229-1345"] = {"U2M": ("20181230-1219_gazenet_u2m_augmented", "gazenet_u2m_augmented")}
    # SIMPLISTIC GAN
    # TODO
    # models["20190105-1325"] = {"U2M": ("20190108-1308_gazenet_u2m_augmented_bw", "gazenet_u2m_augmented_bw")}
    # EGE GAN
    # models["20190120-1420_ege_l1"] = {"U2M": ()}
    # models["20190112-1740_ege_l5"] = {"U2M": ()}
    # models["20190113-1455_ege_l8"] = {"U2M": ("20190114-2346_gazenet_u2m_bw_ege_l8", "gazenet_u2m_bw_ege_l8")}
    # models["20190114-0959_ege_l15"] = {"U2M": ()}
    # models["20190118-1522_ege_l30"] = {"U2M": ()}
    # models["20190118-1542_ege_l50"] = {"U2M": ()}
    #
    # models["20190120-1421_ege_l1_id1"] = {"U2M": ()}
    # models["20190115-1856_ege_l15_id10"] = {"U2M": ()}
    # # LM GANs
    # models["20190120-1423_lm_l1"] = {"U2M": ()}
    # models["20190116-2156_lm_l5"] = {"U2M": ()}
    # models["20190117-1430_lm_l8"] = {"U2M": ()}
    # models["20190116-2305_lm_l15"] = {"U2M": ()}
    # #
    # # TODO models["20190120-1424_lm_l1_id1"] = {"U2M": ()}
    # models["20190117-1548_lm_l10_id5"] = {"U2M": ()}

    models["20190122-2251_rhp_ege_30"] = {"U2M": ()}
    models["20190123-0840_rhp_ege_20_id1"] = {"U2M": ()}
    models["20190123-1455_rhp_id2"] = {"U2M": ()}

    for model_identifier in models.keys():
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
                    path_elg=path_elg,
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
                    n_images=15,
                    image_size=(36, 60),
                    norm="batch",
                    normalise_gaze=False,
                    gazenet_name=gazenet_name,
                    sess=sess)
                identifiers = u2m_visualisation.sample_identifiers(
                    path_original, path_refined)
                u2m_visualisation.visualise(identifiers)

