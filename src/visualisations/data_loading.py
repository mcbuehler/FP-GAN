import h5py
import os
import ujson
from scipy import misc
from matplotlib.pyplot import imshow
import matplotlib.pyplot as ply
import numpy as np


class DataLoader:
    def __init__(self, path):
        self.path = path

    def get_data(self, identifiers):
        raise NotImplementedError()


class MPIIDataLoader(DataLoader):
    def sample_identifiers(self, hf):
        samples = list()
        for person_identifier in hf:
            n = hf[person_identifier]['image'].shape[0]
            index = np.random.randint(0, n)
            samples.append((person_identifier, index))
        return samples

    def get_data(self, identifiers):
        """

        :param path:
        :param identifiers: list of tuples (person_identifier, index)
        :return:
        """
        out = dict()
        with h5py.File(self.path, 'r') as hf:
            for i, (person_identifier, index) in enumerate(identifiers):
                out[identifiers[i]] = {
                    'eye': hf[person_identifier]['image'][index][..., ::-1],
                    'gaze': hf[person_identifier]['gaze'][index]
                }
        return out


class RefinedMPIIDataLoader(DataLoader):
    def sample_identifiers(self):
        person_identifiers = list(set([f[:3] for f in os.listdir(self.path)]))
        index = np.random.randint(0, 1000, len(person_identifiers))
        return [(person_identifiers[i], index[i]) for i in range(len(index))]

    def get_data(self, identifiers):
        def get_file_path(path, person_identifier, index, postfix):
            return os.path.join(path, "{}_{}.{}".format(person_identifier, index, postfix))

        out = dict()
        for i, (person_identifier, index) in enumerate(identifiers):
            img = misc.imread(get_file_path(self.path, person_identifier, index, 'jpg'))
            with open(get_file_path(self.path, person_identifier, index, 'json'), 'r') as f:
                json_data = ujson.load(f)
            out[identifiers[i]] = {
                'eye': img,
                'gaze': json_data['gaze']
            }
        return out


class UnityDataLoader(DataLoader):
    def sample_identifiers(self, size=100):
        from util.files import listdir
        identifiers = listdir(self.path, postfix=".jpg", return_postfix=False)
        return np.random.choice(identifiers, size)

    def get_data(self, identifiers):
        from input.preprocessing import UnityPreprocessor

        def get_file_path(path,  index, postfix):
            return os.path.join(path, "{}.{}".format(index, postfix))

        out = dict()
        for i, file_stem in enumerate(identifiers):
            img = misc.imread(get_file_path(self.path, file_stem, 'jpg'))
            with open(get_file_path(self.path, file_stem, 'json'), 'r') as f:
                json_data = ujson.load(f)
            out[identifiers[i]] = {
                'eye': img,
                # TODO: see difference to original gaze
                'gaze': UnityPreprocessor.look_vec_to_gaze_vec(json_data)[0],
                'original_gaze': UnityPreprocessor.look_vec_to_gaze_vec(json_data)[1]
            }
        return out


class RefinedUnityDataLoader(UnityDataLoader):
    def get_data(self, identifiers=None):
        def get_file_path(path,  index, postfix):
            return os.path.join(path, "{}.{}".format(index, postfix))

        out = dict()
        for i, file_stem in enumerate(identifiers):
            img = misc.imread(get_file_path(self.path, file_stem, 'jpg'))
            img_orig = misc.imread(get_file_path(self.path, "{}_clean".format(file_stem), 'jpg'))
            with open(get_file_path(self.path, file_stem, 'json'), 'r') as f:
                json_data = ujson.load(f)
            out[identifiers[i]] = {
                'eye': img,
                'eye_original': img_orig,
                'gaze': json_data['gaze'],
            }
        return out


if __name__ == "__main__":
    file_stems = [1]
    dl = UnityDataLoader('../data/UnityEyes')
    print(dl.get_data(file_stems)[1]['gaze'])
    dl = RefinedUnityDataLoader('../data/refined_Unity2MPII')
    print(dl.get_data(file_stems)[1]['gaze'])
