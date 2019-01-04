import matplotlib.pyplot as plt
from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze


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

if __name__ == "__main__":
    m2u_visualisation = M2UVisualisation()
    # identifiers = [
    #     ('p00', 0), ('p00', 1), ('p00', 10), ('p00', 11)
    # ]
    identifiers = None
    m2u_visualisation.visualise(identifiers)