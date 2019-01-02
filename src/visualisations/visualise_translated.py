import matplotlib.pyplot as plt
from visualisations.data_loading import get_refined_mpii, get_mpii

from util.gaze import draw_gaze


class M2UVisualisation:
    def __init__(self):
        self.path_mpii = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
        self.path_m2u = '../data/refined_MPII2Unity/'
        self.do_draw_gaze = True

    @staticmethod
    def dg(img, gaze):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=100.0, thickness=2, color=(0, 255, 0),
        )

    def get_data(self, identifiers):
        mpii_data = get_mpii(self.path_mpii,
                         identifiers)
        m2u_data = get_refined_mpii(self.path_m2u, identifiers)
        return  mpii_data, m2u_data

    def visualise(self, identifiers):
        N = len(identifiers)
        fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(20, 20))
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

        plt.subplots_adjust(wspace=.05, hspace=.05)
        plt.show()
        plt.title("MPII vs. refined MPII")


if __name__ == "__main__":
    m2u_visualisation = M2UVisualisation()
    identifiers = [
        ('p00', 0), ('p00', 1), ('p00', 10), ('p00', 11)
    ]
    m2u_visualisation.visualise(identifiers)