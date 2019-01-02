import h5py
import os
import ujson
from scipy import misc
from matplotlib.pyplot import imshow
import matplotlib.pyplot as ply


def get_mpii(path, identifiers):
    """

    :param path:
    :param identifiers: list of tuples (person_identifier, index)
    :return:
    """
    out = dict()

    with h5py.File(path, 'r') as hf:
        for i, (person_identifier, index) in enumerate(identifiers):
            out[identifiers[i]] = {
                'eye': hf[person_identifier]['image'][index][..., ::-1],
                'gaze': hf[person_identifier]['gaze'][index]
            }
    return out


def get_refined_mpii(path, identifiers):
    def get_file_path(path, person_identifier, index, postfix):
        return os.path.join(path, "{}_{}.{}".format(person_identifier, index, postfix))

    out = dict()
    for i, (person_identifier, index) in enumerate(identifiers):
        img = misc.imread(get_file_path(path, person_identifier, index, 'jpg'))
        with open(get_file_path(path, person_identifier, index, 'json'), 'r') as f:
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
