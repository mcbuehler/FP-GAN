import matplotlib.pyplot as plt
from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, UnityDataLoader, RefinedUnityDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze
import os


class M2UVisualisation:
    def __init__(self):
        self.path_mpii = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
        self.path_m2u = '../data/refined_MPII2Unity/'
        self.dl_mpii = MPIIDataLoader(self.path_mpii)
        self.dl_refined_mpii = RefinedMPIIDataLoader(self.path_m2u)
        self.do_draw_gaze = True

    def sample_identifiers(self):
        samples = list()
        with h5py.File(self.path_mpii, 'r') as hf:
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
        mpii_data = self.dl_mpii.get_data(identifiers)
        m2u_data = self.dl_refined_mpii.get_data(identifiers)
        return  mpii_data, m2u_data

    def visualise(self, identifiers=None):
        if identifiers is None:
            identifiers = self.sample_identifiers()
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(20, 20))

        # Axis labels
        cols = ["MPII (R)", "Refined MPII (S')"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')
        for ax, row in zip(axes[:, 1], identifiers):
            ax.text(s=row.__str__(), x=0.5, y=0.9, size='large')

        mpii_data, m2u_data = self.get_data(identifiers)
        for i, (person_identifier, img_index) in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_mpii = mpii_data[(person_identifier, img_index)]['eye']
            img_m2u = m2u_data[(person_identifier, img_index)]['eye']
            if self.do_draw_gaze:
                img_mpii = self.dg(img_mpii, mpii_data[(person_identifier, img_index)]['gaze'])
                img_m2u = self.dg(img_m2u, m2u_data[(person_identifier, img_index)]['gaze'])

            axes[row, 0].imshow(img_mpii)
            axes[row, 1].imshow(img_m2u)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig('../visualisations/mpii_vs_refined-mpii.png')

        g_mpii = [mpii_data[(person_identifier, img_index)]['gaze'] for
                  person_identifier, img_index in identifiers]
        g_m2u = [m2u_data[(person_identifier, img_index)]['gaze'] for
                 person_identifier, img_index in identifiers]


class U2MVisualisation:
    def __init__(self):
        self.path_original = '../data/UnityEyes'
        self.path_refined = '../data/refined_Unity2MPII/'
        self.dl_original = UnityDataLoader(self.path_original)
        self.dl_refined_mpii = RefinedUnityDataLoader(self.path_refined)
        self.do_draw_gaze = True

    def sample_identifiers(self):
        from util.files import listdir
        file_stems = listdir(self.path_original, postfix='.jpg',
                             return_postfix=False)
        index = list()
        c = 0
        while c < 15:
            sample = np.random.choice(file_stems)
            if os.path.exists(os.path.join(self.path_refined, "{}.jpg".format(sample))):
                index.append(sample)
                c += 1
        return index

    @staticmethod
    def dg(img, gaze, length=100):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=length, thickness=2, color=(0, 255, 0),
        )

    def get_data(self, identifiers=None):
        return self.dl_original.get_data(identifiers), self.dl_refined_mpii.get_data(identifiers)

    def visualise(self, identifiers=None):
        if identifiers is None:
            identifiers = self.sample_identifiers()
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

        mpii_data, m2u_data = self.get_data(identifiers)
        for i, file_stem in enumerate(identifiers):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_original_full = mpii_data[file_stem]['eye']
            img_original = m2u_data[file_stem]['eye_original']
            img_refined = m2u_data[file_stem]['eye']
            if self.do_draw_gaze:
                img_original_full = self.dg(img_original_full, mpii_data[file_stem]['gaze'], length=200)
                img_original = self.dg(img_original, mpii_data[file_stem]['gaze'])
                img_refined = self.dg(img_refined, m2u_data[file_stem]['gaze'], length=50)

            axes[row, 0].imshow(img_original_full)
            axes[row, 1].imshow(img_original)
            axes[row, 2].imshow(img_refined)

        plt.subplots_adjust(wspace=.0005, hspace=0.0001, bottom=0, top=0.95)

        # plt.show()
        plt.savefig('../visualisations/unity_vs_refined-unity.png')

if __name__ == "__main__":
    m2u_visualisation = M2UVisualisation()
    # identifiers = [
    #     ('p00', 0), ('p00', 1), ('p00', 10), ('p00', 11)
    # ]
    identifiers = None
    # m2u_visualisation.visualise(identifiers)

    u2m_visualisation = U2MVisualisation()
    u2m_visualisation.visualise(identifiers)