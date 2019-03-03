import matplotlib.pyplot as plt
from visualisations.data_loading import MPIIDataLoader, RefinedMPIIDataLoader, UnityDataLoader, RefinedUnityDataLoader
import h5py
import numpy as np
from util.gaze import draw_gaze
import os


class CompareVisualisation:
    @staticmethod
    def sample_identifiers_mpii(path_mpii):
        samples = list()
        with h5py.File(path_mpii, 'r') as hf:
            for person_identifier in hf:
                n = hf[person_identifier]['image'].shape[0]
                index = np.random.randint(0, n)
                samples.append((person_identifier, index))
        return samples

    def find_similary_unity(self, gaze, data_u):
        best_i = None
        delta = 10000
        for key, elem in data_u.items():

            delta_tmp = np.sum(np.abs(elem['gaze'] - gaze))
            if delta_tmp < delta:
                delta = delta_tmp
                best_i = key
        return data_u[best_i]


    @staticmethod
    def dg(img, gaze):
        return draw_gaze(
            img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            gaze, length=100.0, thickness=2, color=(0, 255, 0),
        )

    def visualise(self, pairs):
        N = len(pairs)
        fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(20, 20))

        # Axis labels
        cols = ["MPII (R)", "Unity (S)"]
        for ax, col in zip(axes[0], cols):
            # Column title
            ax.set_title(col)

        for i, (elem_m, elem_u) in enumerate(pairs):
            row = i
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            img_m = elem_m['eye']
            img_u = elem_u['eye']
            #if self.do_draw_gaze:
            if True:
                img_m = self.dg(img_m, elem_m['gaze'])
                img_u = self.dg(img_u, elem_u['gaze'])

            axes[row, 0].imshow(img_m)
            axes[row, 1].imshow(img_u)


        plt.show()
        # plt.savefig(os.path.join('../visualisations/', self.name_out))


if __name__ == "__main__":
    from input.preprocessing import UnityPreprocessor
    import cv2 as cv
    path_mpii = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
    path_unity = '../data/UnityEyes'

    dl_unity = UnityDataLoader(path_unity)
    identifiers_u = dl_unity.sample_identifiers(100)
    data_u = dl_unity.get_data(identifiers_u)

    dl_mpii = MPIIDataLoader(path_mpii)
    random_p = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06']
    random_i = np.random.randint(0, 2000, len(random_p))
    data_m = dl_mpii.get_data(list(zip(random_p, random_i)))

    vis = CompareVisualisation()
    pairs = list()
    #preprocessor_u = UnityPreprocessor(do_augmentation=False, eye_image_shape=(36, 60))
    for key in data_m.keys():
        match = vis.find_similary_unity(data_u=data_u, gaze=data_m[key]['gaze'])
        #match = preprocessor_u.preprocess(match['eye'], match)
        u = match['eye']
        match['eye'] = u[190:-190,190:-190]
        #match['eye'] = cv.resize(match['eye'], dsize=(120,72), interpolation=cv.INTER_CUBIC)
        pairs.append((data_m[key], match))
    vis.visualise(pairs)






