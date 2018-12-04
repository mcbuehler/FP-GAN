import h5py
import tensorflow as tf
from input.preprocessing import MPIIPreprocessor


class MPIIGenerator:
    def __init__(self, file, shuffle=False):
        self.file = file
        self.eye_shape = self._get_eye_shape()
        # tuple with (person_identifier, index) for all people and
        # number of samples per person
        # e.g. ('p01', 5) refers to sample 5 for person 'p01'
        self.all_identifiers = self._create_all_identifiers()
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


class MPIIDataset:
    """
    Dataset for MPIIGaze data.
    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, testing=False):
        self.path_input = path_input
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.testing = testing
        self.preprocessor = MPIIPreprocessor(testing=testing,
                                        eye_image_shape=self.image_size)

    def get_iterator(self):
        generator = MPIIGenerator(self.path_input, shuffle=self.shuffle)
        dataset = tf.data.Dataset.from_generator(
            generator,
            {'eye': tf.uint8, 'gaze': tf.float32},
            {'eye': tf.TensorShape([*generator.eye_shape, 3]), 'gaze': tf.TensorShape([2])}
            )

        dataset = dataset.map(self._get_tensors)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
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
        return {'eye': eye_preprocessed, 'gaze': entry['gaze']}


if __name__ == "__main__":
    path_input = '../data/MPIIFaceGaze/single-eye_zhang.h5'

    dataset = MPIIDataset(path_input, batch_size=10, image_size=(72, 120))
    iterator = dataset.get_iterator()

    with tf.Session() as sess:
        import numpy as np
        gazes = np.empty((1,2))
        # n_batches = int(dataset.N / dataset.batch_size)
        for i in range(10):
            # print(i, "/", n_batches)

            next_element = iterator.get_next()
            elem = sess.run(next_element)
            print(elem['eye'].shape)
            # imshow(elem['eye'][0])
            # ply.show()
            # gazes = np.concatenate([gazes, elem['gaze']], 0)

