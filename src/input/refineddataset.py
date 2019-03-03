import os

import tensorflow as tf

from input.base_dataset import BaseDataset
from input.preprocessing import RefinedPreprocessor
from util.files import listdir


class RefinedDataset(BaseDataset):
    """
    Dataset for Refined images (both Unity and MPII).
    Call get_iterator() to get an iterator for this dataset.

    Example usage:

    path_input = '../data/refined_Unity2MPII_Train/'

    dataset = RefinedDataset(path_input=path_input, batch_size=1,
                             image_size=(72, 120),
                             do_augmentation=False, rgb=False)
    iterator = dataset.get_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        elem = sess.run(next_element)
        print(elem)
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = RefinedPreprocessor(
            do_augmentation=kwargs['do_augmentation'],
            eye_image_shape=self.image_size)

    def _read_and_preprocess(self, file_stem, path, preprocessor):
        file_path_no_prefix = os.path.join(path, file_stem.decode('utf-8'))
        image = self._read_image("{}.jpg".format(file_path_no_prefix))
        json_data = self._read_json("{}.json".format(file_path_no_prefix))
        result_tensors = preprocessor.preprocess(image, json_data)
        return result_tensors

    def _get_filestems_tensor(self):
        file_stems = listdir(self.path_input, postfix=".jpg",
                             return_postfix=False)
        file_stems = filter(lambda f: "_clean" not in f, file_stems)
        file_stems = list(set(file_stems))
        self.N = len(file_stems)
        file_stems = tf.constant(file_stems, dtype=tf.string,
                                 name="file_stems")
        return file_stems

    def get_iterator(self):
        """
        Each element returned by the iterator has the following keys:
            'id', 'clean_eye', 'eye', 'gaze'
        Returns: iterator for this dataset.
        """
        file_stems = self._get_filestems_tensor()
        dataset = tf.data.Dataset.from_tensor_slices(file_stems)
        dataset = dataset.map(self._get_tensors,
                              num_parallel_calls=self.num_parallel_calls)

        iterator = self._prepare_iterator(dataset)
        self._iterator_ready_info()
        return iterator

    def _get_tensors(self, file_stem):
        clean_eye, eye, gaze = tf.py_func(lambda file_stem:
                                          self._read_and_preprocess(
                                              file_stem, self.path_input,
                                              self.preprocessor),
                                          [file_stem],
                                          Tout=[tf.float32, tf.float32,
                                                tf.float32]
                                          )
        # We need to set shapes because we need to know them when we
        # build the execution graph (images only).
        # The output tensor does not need a shape at this point.
        if self.rgb:
            image_shape = (*self.image_size, 3)
        else:
            image_shape = (*self.image_size, 1)
            clean_eye, eye = self._expand_dims(clean_eye, eye)

        clean_eye.set_shape(image_shape)
        eye.set_shape(image_shape)
        return {'id': file_stem, 'clean_eye': clean_eye, 'eye': eye,
                'gaze': gaze}
