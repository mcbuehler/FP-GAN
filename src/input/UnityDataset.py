import os
import ujson
import cv2 as cv
import tensorflow as tf

from input.preprocessing import ImagePreprocessor
from util.files import listdir


class UnityDataset:
    """
    Dataset for UnityEyes.
    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True):
        self.path_input = path_input
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.unity_preprocessor = ImagePreprocessor(testing=False,
                                               eye_image_shape=self.image_size)

    def _read_image(self, filename):
        image = cv.imread(filename, cv.IMREAD_COLOR)
        # CV loads the image as BGR
        # Convert image to RGB
        image = image[..., ::-1]
        return image

    def _read_json(self, filename):
        # print(filename)
        with open(filename, 'r') as f:
            json_data = ujson.load(f)
        return json_data

    def _read_and_preprocess(self, file_stem, path, preprocessor):
        file_path_no_prefix = os.path.join(path, file_stem.decode('utf-8'))
        image = self._read_image("{}.jpg".format(file_path_no_prefix))
        json_data = self._read_json("{}.json".format(file_path_no_prefix))
        image = preprocessor.preprocess(image, json_data)
        return image

    def _get_filestems_tensor(self):
        file_stems = listdir(self.path_input, postfix=".jpg", return_postfix=False)
        self.N = len(file_stems)
        print('N:', self.N)
        file_stems = tf.constant(file_stems, dtype=tf.string,
                                 name="file_stems")
        return file_stems

    def get_iterator(self):
        file_stems = self._get_filestems_tensor()
        dataset = tf.data.Dataset.from_tensor_slices(file_stems)
        dataset = dataset.map(self._get_tensors)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def _get_tensors(self, file_stem):
        clean_eye, eye, gaze = tf.py_func(lambda file_stem:
                                          self._read_and_preprocess(
                                              file_stem, path_input,
                                              self.unity_preprocessor),
                                          [file_stem],
                                          Tout=[tf.float32, tf.float32,
                                                tf.float32]
                                          )
        return {'clean_eye': clean_eye, 'eye': eye, 'gaze': gaze}


if __name__ == "__main__":
    path_input = '../data/UnityEyesTest/'

    tf.enable_eager_execution()
    dataset = UnityDataset(path_input, batch_size=10, image_size=(72, 120))
    iterator = dataset.get_iterator()

    n_batches = int(dataset.N / dataset.batch_size)
    for i in range(n_batches+3):
        print(i, "/", n_batches)
        try:
            next_element = iterator.get_next()
        except Exception as e:
            print(e)
            print("Value Error. Skipping.")
