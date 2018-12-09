import os
import ujson
import cv2 as cv
import tensorflow as tf

from input.base_dataset import BaseDataset

from input.preprocessing import UnityPreprocessor
from util.files import listdir


class UnityDataset(BaseDataset):
    """
    Dataset for UnityEyes.
    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, testing=False, repeat=True, drop_remainder=False):
        super().__init__(path_input, image_size, batch_size, shuffle, buffer_size, testing, repeat, drop_remainder=drop_remainder)

        self.unity_preprocessor = UnityPreprocessor(testing=testing,
                                                    eye_image_shape=self.image_size)

    def _read_image(self, filename):
        image = cv.imread(filename, cv.IMREAD_COLOR)
        # CV loads the image as BGR
        # Convert image to RGB
        image = image[..., ::-1]
        return image

    def _read_json(self, filename):
        with open(filename, 'r') as f:
            json_data = ujson.load(f)
        return json_data

    def _read_and_preprocess(self, file_stem, path, preprocessor):
        file_path_no_prefix = os.path.join(path, file_stem.decode('utf-8'))
        image = self._read_image("{}.jpg".format(file_path_no_prefix))
        json_data = self._read_json("{}.json".format(file_path_no_prefix))
        result_tensors = preprocessor.preprocess(image, json_data)
        return result_tensors

    def _get_filestems_tensor(self):
        file_stems = listdir(self.path_input, postfix=".jpg", return_postfix=False)
        self.N = len(file_stems)
        file_stems = tf.constant(file_stems, dtype=tf.string,
                                 name="file_stems")
        return file_stems

    def get_iterator(self):
        file_stems = self._get_filestems_tensor()
        dataset = tf.data.Dataset.from_tensor_slices(file_stems)
        dataset = dataset.map(self._get_tensors, num_parallel_calls=self.num_parallel_calls)

        iterator = self._prepare_iterator(dataset)
        self._iterator_ready_info()
        return iterator

    def _get_tensors(self, file_stem):
        clean_eye, eye, gaze = tf.py_func(lambda file_stem:
                                          self._read_and_preprocess(
                                              file_stem, self.path_input,
                                              self.unity_preprocessor),
                                          [file_stem],
                                          Tout=[tf.float32, tf.float32,
                                                tf.float32]
                                          )
        # We need to set shapes because we need to know them when we
        # build the execution graph (images only).
        # The output tensor does not need a shape at this point.
        image_shape = (*self.image_size, 3)
        clean_eye.set_shape(image_shape)
        eye.set_shape(image_shape)
        return {'id': file_stem, 'clean_eye': clean_eye, 'eye': eye, 'gaze': gaze}


if __name__ == "__main__":
    path_input = '../data/UnityEyesTest/'

    dataset = UnityDataset(path_input, batch_size=10, image_size=(72, 120))
    iterator = dataset.get_iterator()

    with tf.Session() as sess:
        n_batches = int(dataset.N / dataset.batch_size)
        for i in range(n_batches+3):
            print(i, "/", n_batches)
            try:
                next_element = iterator.get_next()
                print(sess.run(next_element['clean_eye']).shape)
            except Exception as e:
                print("Value Error. Skipping.")
