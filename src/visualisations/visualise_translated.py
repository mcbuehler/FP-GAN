import matplotlib.pyplot as plt
from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, UnityDataLoader, RefinedUnityDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze
import os


class Visualisation:
    def __init__(self,
                 dl_original,
                 dl_refined,
                 name_out='mpii_vs_refined-mpii.png',
                 do_draw_gaze=True):
        self.name_out = name_out
        self.dl_original = dl_original
        self.dl_refined = dl_refined
        self.do_draw_gaze = do_draw_gaze

    def visualise(self, identifiers):
        raise NotImplementedError()


class M2UVisualisation(Visualisation):

    @staticmethod
    def sample_identifiers(path_original):
        samples = list()
        with h5py.File(path_original, 'r') as hf:
            for person_identifier in hf:
                n = hf[person_identifier]['image'].shape[0]
                index = np.random.randint(0, n)
                samples.append((person_identifier, index))
        return samples

    @staticmethod
    def dg(img, gaze):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=100.0, thickness=2, color=(0, 255, 0),
        )

    def get_data(self, identifiers=None):
        mpii_data = self.dl_original.get_data(identifiers)
        m2u_data = self.dl_refined.get_data(identifiers)
        return  mpii_data, m2u_data

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

        mpii_data, m2u_data = self.get_data(identifiers)
        for i, (person_identifier, img_index) in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original = mpii_data[(person_identifier, img_index)]['eye']
            img_refined = m2u_data[(person_identifier, img_index)]['eye']
            if self.do_draw_gaze:
                img_original = self.dg(img_original, mpii_data[(person_identifier, img_index)]['gaze'])
                img_refined = self.dg(img_refined, m2u_data[(person_identifier, img_index)]['gaze'])

            if len(img_refined.shape) == 2:
                # We have a b/w image
                cmap = 'gray'
            else:
                cmap = None
            axes[row, 0].imshow(img_original, cmap=cmap)
            axes[row, 1].imshow(img_refined, cmap=cmap)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig(os.path.join('../visualisations/', self.name_out))


class U2MVisualisation(Visualisation):

    @staticmethod
    def sample_identifiers(path_original, path_refined):
        from util.files import listdir
        file_stems = listdir(path_original, postfix='.jpg',
                             return_postfix=False)
        index = list()
        c = 0
        while c < 15:
            sample = np.random.choice(file_stems)
            if os.path.exists(os.path.join(path_refined, "{}.jpg".format(sample))):
                index.append(sample)
                c += 1
        return index

    @staticmethod
    def dg(img, gaze, length=100, thickness=2):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=length, thickness=thickness, color=(0, 255, 0),
        )

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
        for i, file_stem in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original_full = original_data[file_stem]['eye']
            img_original = refined_data[file_stem]['eye_original']
            img_refined = refined_data[file_stem]['eye']
            if self.do_draw_gaze:
                img_original_full = self.dg(img_original_full, original_data[file_stem]['gaze'], length=400, thickness=5)
                img_original = self.dg(img_original, original_data[file_stem]['gaze'])
                img_refined = self.dg(img_refined, refined_data[file_stem]['gaze'], length=100)

            if len(img_refined.shape) == 2:
                # We have a b/w image
                cmap = 'gray'
            else:
                cmap = None
            axes[row, 0].imshow(img_original_full, cmap=cmap)
            axes[row, 1].imshow(img_original, cmap=cmap)
            axes[row, 2].imshow(img_refined, cmap=cmap)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig(os.path.join('../visualisations/', self.name_out))


if __name__ == "__main__":
    M2U = True
    U2M = True

    if M2U:
        path_original = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
        path_refined = '../data/refined_MPII2Unity/'
        dl_original = MPIIDataLoader(path_original)
        dl_refined = RefinedMPIIDataLoader(path_refined)

        m2u_visualisation = M2UVisualisation(
            dl_original=dl_original,
            dl_refined=dl_refined,
            name_out='mpii_vs_refined_bw.png')
        identifiers = M2UVisualisation.sample_identifiers(path_original)
        m2u_visualisation.visualise(identifiers)

    if U2M:
        path_original = '../data/UnityEyes'
        path_refined = '../data/refined_Unity2MPII/'
        dl_original = UnityDataLoader(path_original)
        dl_refined = RefinedUnityDataLoader(path_refined)

        identifiers = U2MVisualisation.sample_identifiers(path_original, path_refined)

        u2m_visualisation = U2MVisualisation(
            dl_original=dl_original,
            dl_refined=dl_refined,
            name_out='unity_vs_refined_bw.png')
        u2m_visualisation.visualise(identifiers)