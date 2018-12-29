import os
import ujson
import cv2 as cv
import re

import numpy as np
import tensorflow as tf

from input.base_dataset import BaseDataset

from input.preprocessing import RefinedPreprocessor
from util.files import listdir


class RefinedDataset(BaseDataset):
    """
    Dataset for Refined images (both Unity and MPII).
    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, do_augmentation=False, repeat=True, drop_remainder=False):
        super().__init__(path_input, image_size, batch_size, shuffle, buffer_size, do_augmentation, repeat, drop_remainder=drop_remainder)

        self.preprocessor = RefinedPreprocessor(do_augmentation=do_augmentation,
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
        # If we are dealing with refined images, we migth have a postfix "_clean"
        # But we are only interested in the image ids.
        # r'(p\d\d_\d+).jpg'
        # for f in files:
        #     match = re.findall(self.id_pattern, f)
        #     if match:
        #         ids.append(match[0])
        # TODO: maybe we need to remove the files that have _clean in the filename
        # file_stems = list(set([re.sub(r'[^0-9]+', '', f) for f in file_stems]))
        file_stems = filter(lambda f: "_clean" not in f, file_stems)
        file_stems = list(set(file_stems))
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
                                              self.preprocessor),
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
    path_input = '../data/refined_Unity2MPII_Train/'
    #path_input = '../data/refined_MPII2Unity_Train/'

    dataset = RefinedDataset(path_input, batch_size=10, image_size=(72, 120),
                             do_augmentation=False)
    iterator = dataset.get_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        n_batches = int(dataset.N / dataset.batch_size)
        for i in range(n_batches+3):
            print(i, "/", n_batches)
            try:
                elem = sess.run(next_element)

                from matplotlib.pyplot import imshow
                from util.gaze import draw_gaze
                import matplotlib.pyplot as plt
                for j in range(30):

                    img = np.array((elem['eye'][j]+1) * 128,  dtype=np.int)
                    # img = elem['eye'][j]
                    print(img)
                    gaze = elem['gaze'][j]

                    img = draw_gaze(
                        img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
                        gaze, length=100.0, thickness=2, color=(0, 255, 0),
                    )
                    imshow(img)
                    plt.title("Gaze: {:.3f} {:.3f}".format(*elem['gaze'][j]))
                    plt.show()
            except Exception as e:
                print("Value Error. Skipping.")
