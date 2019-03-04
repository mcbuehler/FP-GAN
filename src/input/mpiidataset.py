import h5py
import numpy as np
import tensorflow as tf

from input.base_dataset import BaseDataset
from input.preprocessing import MPIIPreprocessor


class MPIIGenerator:
    """
    Generator for loading data from an MPIIFaceGaze h5 File
    """

    def __init__(self, file, shuffle=False, rgb=True):
        self.file = file
        self.eye_shape = self._get_eye_shape()
        # tuple with (person_identifier, index) for all people and
        # number of samples per person
        # e.g. ('p01', 5) refers to sample 5 for person 'p01'
        self.all_identifiers = self._create_all_identifiers()
        self.N = len(self.all_identifiers)
        self.rgb = rgb
        if shuffle:
            # We want to draw samples from different people in random order
            self.all_identifiers = list(set(self.all_identifiers))

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for person_identifier, index in self.all_identifiers:
                clean_eye = hf[person_identifier]['image'][index]
                eye = hf[person_identifier]['image'][index]
                gaze = hf[person_identifier]['gaze'][index]

                if not self.rgb:
                    # convert rgb to b/w
                    clean_eye = np.mean(hf[person_identifier]['image'][index],
                                        axis=2)
                    eye = np.mean(hf[person_identifier]['image'][index],
                                  axis=2)

                yield {
                    'eye': eye,
                    # For MPII we only have clean eyes for the moment
                    'clean_eye': clean_eye,
                    'gaze': gaze,
                    # 'landmarks': hf[person_identifier]['landmarks'][index],
                    'head': hf[person_identifier]['head'][index],
                    'id': [self._create_single_identifier(person_identifier,
                                                          index)]
                }

    @staticmethod
    def _create_single_identifier(person, index):
        return "{}_{}".format(person, index)

    def _create_all_identifiers(self):
        identifiers = list()

        with h5py.File(self.file, 'r') as hf:
            person_identifiers = hf.keys()
            for person_identifier in person_identifiers:
                n_entries = hf[person_identifier]['gaze'].shape[0]
                identifiers += [(person_identifier, i) for i in
                                range(n_entries)]
        return identifiers

    def _get_eye_shape(self):
        with h5py.File(self.file, 'r') as hf:
            eye_shape = hf[list(hf.keys())[0]]['image'].shape[1:3]
        return eye_shape


class MPIIDataset(BaseDataset):
    """
    Dataset for MPIIGaze data.
    Call get_iterator() to get an iterator for this dataset.

    Example usage:
    path_input = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'
    dataset = MPIIDataset(path_input=path_input, batch_size=1000, shuffle=True,
                          image_size=(72, 120), rgb=False,
                          normalise_gaze=False)
    iterator = dataset.get_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        elem = sess.run(next_element['gaze'])
        print(elem)
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = MPIIPreprocessor(eye_image_shape=self.image_size)

    def get_iterator(self, repeat=True):
        """
        Each element returned by the iterator has the following keys:
            'id', 'eye', 'clean_eye', 'gaze', 'head'
        Returns: iterator for this dataset.
        """
        generator = MPIIGenerator(self.path_input, shuffle=self.shuffle,
                                  rgb=self.rgb)
        self.N = generator.N
        image_shape = [*generator.eye_shape,
                       3] if self.rgb else generator.eye_shape

        dataset = tf.data.Dataset.from_generator(
            generator,
            {
                'id': tf.string,
                'eye': tf.uint8,
                'clean_eye': tf.uint8,
                'gaze': tf.float32,
                # 'landmarks': tf.float32,
                'head': tf.float32
            },
            {
                'id': tf.TensorShape(1),
                'eye': tf.TensorShape(image_shape),
                'clean_eye': tf.TensorShape(image_shape),
                'gaze': tf.TensorShape([2]),
                # 'landmarks': tf.TensorShape([18, 2]),
                'head': tf.TensorShape([2])
            }
        )

        dataset = dataset.map(self._get_tensors,
                              num_parallel_calls=self.num_parallel_calls)
        iterator = self._prepare_iterator(dataset)

        self._iterator_ready_info()
        return iterator

    def _get_tensors(self, entry):
        eye = tf.py_func(lambda image:
                         self.preprocessor.preprocess(image),
                         [entry['eye']],
                         Tout=[tf.float32]
                         )[0]
        # We need to set shapes because we need to know them when we
        # build the execution graph (images only).
        # The output tensor does not need a shape at this point.
        clean_eye = eye

        if self.rgb:
            image_shape = (*self.image_size, 3)
        else:
            image_shape = (*self.image_size, 1)
            clean_eye, eye = self._expand_dims(clean_eye, eye)

        eye.set_shape(image_shape)
        return {
            'eye': eye,
            'clean_eye': eye,
            'gaze': entry['gaze'],
            # 'landmarks': entry['landmarks'],
            'head': entry['head'],
            'id': entry['id']
        }
