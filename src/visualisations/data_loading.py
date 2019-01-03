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

    def get_data(self, identifiers=None):
        raise NotImplementedError()


class MPIIDataLoader(DataLoader):
    def sample_identifiers(self, hf):
        samples = list()
        for person_identifier in hf:
            n = hf[person_identifier]['image'].shape[0]
            index = np.random.randint(0, n)
            samples.append((person_identifier, index))
        return samples

    def get_data(self, identifiers=None):
        """

        :param path:
        :param identifiers: list of tuples (person_identifier, index)
        :return:
        """
        out = dict()

        with h5py.File(self.path, 'r') as hf:
            identifiers = self.sample_identifiers(hf) if identifiers is None else identifiers
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

    def get_data(self, identifiers=None):
        def get_file_path(path, person_identifier, index, postfix):
            return os.path.join(path, "{}_{}.{}".format(person_identifier, index, postfix))

        out = dict()
        identifiers = self.sample_identifiers() if identifiers is None else identifiers
        for i, (person_identifier, index) in enumerate(identifiers):
            img = misc.imread(get_file_path(self.path, person_identifier, index, 'jpg'))
            with open(get_file_path(self.path, person_identifier, index, 'json'), 'r') as f:
                json_data = ujson.load(f)
            out[identifiers[i]] = {
                'eye': img,
                'gaze': json_data['gaze']
            }
        return out


if __name__ == "__main__":
    identifiers = [('p00', 0), ('p00', 10)]
    # mpii_data = get_mpii('../data/MPIIFaceGaze/single-eye-right_zhang.h5', identifiers)
    get_refined_mpii('../data/refined_MPII2Unity/', identifiers)
