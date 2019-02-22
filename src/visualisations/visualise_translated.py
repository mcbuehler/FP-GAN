import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

from input.preprocessing import RefinedPreprocessor
from models.elg import ELGInference
from models.gazenet import GazeNetInference
from util.files import listdir
from util.gaze import draw_gaze, draw_gaze_py, angular_error, euclidean_error, \
    mse


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
                 do_draw_gaze=True,
                 do_predict_gaze=True,
                 do_draw_landmarks=False,
                 do_plot_infobox=True):
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

        self.figsize = 100

        self.color_predicted = (255, 255, 0)
        self.color_true = (255, 0, 0)
        self.color_lm_refined = (0, 255, 255)

        self.n_images = n_images
        self.do_draw_gaze = do_draw_gaze
        self.do_predict_gaze = do_predict_gaze
        self.do_draw_landmarks = do_draw_landmarks
        self.do_plot_infobox = do_plot_infobox

        if path_ege and gazenet_name:
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
        # We want to return a gray-scale image with an explicit channel
        images_preprocessed = [np.expand_dims(img, axis=2) for img in
                               images_preprocessed]
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
    def dg_py(img, gaze, color, length=40, thickness=2):
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
            img = cv2.drawMarker(img, tuple(c), color=color, markerType=2,
                                 markerSize=1)
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

    def save_fig(self, plt):
        path_out = os.path.join('../visualisations/', self.name_out)
        print("Saving figure to {}...".format(path_out))
        plt.savefig(path_out, transparent=True)


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
        print("Visualising {} samples...".format(N))
        ncols = 3 if self.do_plot_infobox else 2
        fig, axes = plt.subplots(nrows=N, ncols=ncols,
                                 figsize=(self.figsize, self.figsize))

        # Axis labels
        cols = ["MPII (R)", "Refined MPII (S')"]
        for ax, col in zip(axes[0], cols):
            # Column title
            ax.set_title(col)

        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)

        if self.do_predict_gaze:
            gaze_true = np.array([original_data[
                                      (pi, ii)][
                                      'gaze'] for pi, ii in identifiers])
            gaze_pred, gaze_error, eucl_gaze_error, ms_error = self._gaze_error(
                images_preprocessed,
                gaze_true
            )

        # Sort by ascending prediction error (if applicable)
        if self.do_predict_gaze:
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

            if self.do_draw_landmarks:
                landmarks_pred_refined = self._predict_landmarks(
                    images_preprocessed)
                self.draw_landmarks(img_refined, landmarks_pred_refined[i],
                                    self.color_lm_refined)

            if self.do_draw_gaze:
                img_original = self.dg_py(img_original, original_data[
                    (person_identifier, img_index)]['gaze'],
                                          color=self.color_true)
                img_refined = self.dg_py(img_refined, refined_data[
                    (person_identifier, img_index)]['gaze'],
                                         color=self.color_true)
                info_txt = "{} \n true pitch / yaw: {}".format(info_txt,
                                                               self.format_gaze(
                                                                   original_data[
                                                                       (
                                                                       person_identifier,
                                                                       img_index)][
                                                                       'gaze']))
                if self.do_predict_gaze:
                    # img_refined = self.dg(img_refined, gaze_pred[i], color=self.color_predicted)
                    img_refined = self.dg_py(img_refined, gaze_pred[i],
                                             color=self.color_predicted)
                    info_txt = "{} \n predicted pitch / yaw: {} " \
                               "\n error angular / euclidean / mse: {:.2f} / {:.2f} / {:.2f}". \
                        format(info_txt, self.format_gaze(gaze_pred[i]),
                               gaze_error[i], eucl_gaze_error[i], ms_error[i])

            axes[row, 0].imshow(img_original)
            axes[row, 1].imshow(img_refined)
            if self.do_plot_infobox:
                TextBox(axes[row, 2], person_identifier, initial=info_txt)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        if show:
            plt.show()
        if save:
            self.save_fig(plt, transparent=True)
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
            if os.path.exists(
                    os.path.join(path_refined, "{}.jpg".format(sample))):
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
        return self.dl_original.get_data(
            identifiers), self.dl_refined.get_data(identifiers)

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
        print("Visualising {} samples...".format(N))
        ncols = 4 if self.do_plot_infobox else 3
        fig, axes = plt.subplots(nrows=N, ncols=ncols,
                                 figsize=(self.figsize, self.figsize))

        # Axis labels
        cols = ["Unity (S) full", "Unity (S) cropped", "Refined Unity (R')"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        original_data, refined_data = self.get_data(identifiers)

        # Get the predictions for translated images
        images_refined = [refined_data[key]['eye'] for key in identifiers]
        images_preprocessed = self.preprocess(images_refined)
        if self.do_predict_gaze:
            gaze_true = np.array([original_data[i][
                                      'gaze'] for i in identifiers])
            gaze_pred, gaze_error, eucl_gaze_error, ms_error = self._gaze_error(
                images_preprocessed,
                gaze_true
            )

        # Sort by ascending prediction error (if applicable)
        if self.do_predict_gaze:
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

            if self.do_draw_landmarks:
                landmarks_pred_refined = self._predict_landmarks(
                    images_preprocessed)
                self.draw_landmarks(img_refined, landmarks_pred_refined[i],
                                    self.color_lm_refined)

            if self.do_draw_gaze:
                # img_original_full = self.dg_py(img_original_full,
                # original_data[file_stem]['gaze'], length=400, thickness=5,
                # color=self.color_true)
                img_original = self.dg_py(img_original,
                                          original_data[file_stem]['gaze'],
                                          color=self.color_true)
                img_refined = self.dg_py(img_refined,
                                         refined_data[file_stem]['gaze'],
                                         color=self.color_true)
                info_txt = "{} \n true pitch / yaw: {}".format(info_txt,
                                                               self.format_gaze(
                                                                   original_data[
                                                                       file_stem][
                                                                       'gaze']))
                if self.do_predict_gaze:
                    img_refined = self.dg_py(img_refined, gaze_pred[i],
                                             color=self.color_predicted)
                    info_txt = "{} \n predicted pitch / yaw: {} " \
                               "\n error angular / euclidean / mse: {:.2f} / {:.2f} / {:.2f}". \
                        format(info_txt, self.format_gaze(gaze_pred[i]),
                               gaze_error[i], eucl_gaze_error[i], ms_error[i])

            axes[row, 0].imshow(img_original_full)
            axes[row, 1].imshow(img_original)
            axes[row, 2].imshow(img_refined)
            if self.do_plot_infobox:
                TextBox(axes[row, 3], file_stem, initial=info_txt)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)
        if show:
            plt.show()
        if save:
            self.save_fig(plt)
        plt.close(fig)
