import h5py
import tensorflow as tf
from input.preprocessing import MPIIPreprocessor
from input.base_dataset import BaseDataset
import numpy as np


class MPIIGenerator:
    def __init__(self, file, shuffle=False):
        self.file = file
        self.eye_shape = self._get_eye_shape()
        # tuple with (person_identifier, index) for all people and
        # number of samples per person
        # e.g. ('p01', 5) refers to sample 5 for person 'p01'
        self.all_identifiers = self._create_all_identifiers()
        self.N = len(self.all_identifiers)
        if shuffle:
            # We want to draw samples from different people in random order
            self.all_identifiers = list(set(self.all_identifiers))

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for person_identifier, index in self.all_identifiers:
                yield {
                    'eye': hf[person_identifier]['image'][index],
                    'gaze': hf[person_identifier]['gaze'][index]
                }

    def _create_all_identifiers(self):
        identifiers = list()

        with h5py.File(self.file, 'r') as hf:
            person_identifiers = hf.keys()
            for person_identifier in person_identifiers:
                n_entries = hf[person_identifier]['gaze'].shape[0]
                identifiers += [(person_identifier, i) for i in range(n_entries)]
        return identifiers

    def _get_eye_shape(self):
        with h5py.File(self.file, 'r') as hf:
            eye_shape = hf[list(hf.keys())[0]]['image'].shape[1:3]
        return eye_shape


class MPIIDataset(BaseDataset):
    """
    Dataset for MPIIGaze data.
    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, testing=False, repeat=True, drop_remainder=False):
        super().__init__(path_input, image_size, batch_size, shuffle, buffer_size, testing, repeat, drop_remainder=drop_remainder)

        self.preprocessor = MPIIPreprocessor(testing=testing,
                                        eye_image_shape=self.image_size)

    def get_iterator(self, repeat=True):
        generator = MPIIGenerator(self.path_input, shuffle=self.shuffle)
        self.N = generator.N

        dataset = tf.data.Dataset.from_generator(
            generator,
            {'eye': tf.uint8, 'gaze': tf.float32},
            {'eye': tf.TensorShape([*generator.eye_shape, 3]),
             'gaze': tf.TensorShape([2])
             }
            )

        dataset = dataset.map(self._get_tensors, num_parallel_calls=self.num_parallel_calls)
        iterator = self._prepare_iterator(dataset)

        self._iterator_ready_info()
        return iterator

    def _get_tensors(self, entry):
        eye_preprocessed = tf.py_func(lambda image:
                                          self.preprocessor.preprocess(image),
                                          [entry['eye']],
                                          Tout=[tf.float32]
                                          )[0]
        # We need to set shapes because we need to know them when we
        # build the execution graph (images only).
        # The output tensor does not need a shape at this point.
        image_shape = (*self.image_size, 3)
        eye_preprocessed.set_shape(image_shape)
        return {
            'eye': eye_preprocessed,
            'gaze': entry['gaze'],
            'landmarks': entry['landmarks'],
            'head': entry['head']
        }


if __name__ == "__main__":
    path_input = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'

    dataset = MPIIDataset(path_input, batch_size=10, image_size=(72, 120))
    iterator = dataset.get_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        # n_batches = int(dataset.N / dataset.batch_size)
        for i in range(10):
            # print(i, "/", n_batches)

            elem = sess.run(next_element)
            print(np.max(elem["gaze"], axis=1))
            print(np.mean(np.max(np.abs(elem["gaze"]), axis=1)))
            print(np.mean(np.mean(np.abs(elem["gaze"]), axis=1)))
            from matplotlib.pyplot import imshow
            from util.gaze import draw_gaze

            import matplotlib.pyplot as plt
            for j in range(10):
                img = np.array((elem['eye'][j]+1) * 128,  dtype=np.int)
                gaze = elem['gaze'][j]

                img = draw_gaze(
                    img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
                    gaze, length=100.0, thickness=2, color=(0, 255, 0),
                )
                imshow(img)
                plt.title("Gaze: {:.3f} {:.3f}".format(*elem['gaze'][j]))
                plt.show()

            # imshow(elem['eye'][0])
            # ply.show()
            # gazes = np.concatenate([gazes, elem['gaze']], 0)

