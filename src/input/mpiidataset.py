import h5py
import tensorflow as tf
from input.preprocessing import MPIIPreprocessor
from input.base_dataset import BaseDataset
import numpy as np


class MPIIGenerator:
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
                if not self.rgb:
                    # convert rgb to b/w
                    clean_eye = np.mean(hf[person_identifier]['image'][index], axis=2)
                    eye = np.mean(hf[person_identifier]['image'][index], axis=2)
                    gaze = hf[person_identifier]['gaze'][index]
                    # We assume that gaze is normalised, but we want it in range [-pi, pi]
                    gaze = np.pi * gaze
                yield {
                    'eye': eye,
                    # For MPII we only have clean eyes for the moment
                    'clean_eye': clean_eye,
                    'gaze': gaze,
                    # 'landmarks': hf[person_identifier]['landmarks'][index],
                    'head': hf[person_identifier]['head'][index],
                    'id': [self._create_single_identifier(person_identifier, index)]
                }

    def _create_single_identifier(self, person, index):
        return "{}_{}".format(person, index)

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # def __init__(self, path_input, image_size=(72, 120), batch_size=32, rgb=True, shuffle=True, buffer_size=1000, do_augmentation=False, repeat=True, drop_remainder=False, filter_gaze=False):
    #     super().__init__(path_input, image_size, batch_size, rgb, shuffle, buffer_size, do_augmentation, repeat, drop_remainder=drop_remainder, filter_gaze=filter_gaze)

        self.preprocessor = MPIIPreprocessor(eye_image_shape=self.image_size)

    def get_iterator(self, repeat=True):
        generator = MPIIGenerator(self.path_input, shuffle=self.shuffle, rgb=self.rgb)
        self.N = generator.N
        image_shape = [*generator.eye_shape, 3] if self.rgb else generator.eye_shape

        dataset = tf.data.Dataset.from_generator(
            generator,
            {'eye': tf.uint8,
             'clean_eye': tf.uint8,
             'gaze': tf.float32,
             # 'landmarks': tf.float32,
             'head': tf.float32,
             'id': tf.string
             },
            {'eye': tf.TensorShape(image_shape),
             'clean_eye': tf.TensorShape(image_shape),
             'gaze': tf.TensorShape([2]),
             # 'landmarks': tf.TensorShape([18, 2]),
             'head': tf.TensorShape([2]),
             'id': tf.TensorShape(1)
             }
            )

        dataset = dataset.map(self._get_tensors, num_parallel_calls=self.num_parallel_calls)
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


if __name__ == "__main__":
    path_input = '../data/MPIIFaceGaze/single-eye-right_zhang.h5'

    dataset = MPIIDataset(path_input=path_input, batch_size=1000, shuffle=True, image_size=(72, 120), rgb=False, normalise_gaze=False)
    from input.unitydataset import UnityDataset
    dataset = UnityDataset(path_input='../data/UnityEyes', batch_size=1000, shuffle=True, image_size=(72, 120), rgb=False, normalise_gaze=True,
                           do_augmentation=False)
    iterator = dataset.get_iterator()
    next_element = iterator.get_next()
    import numpy.random as random
    from util.gaze import angular_error

    with tf.Session() as sess:
        # n_batches = int(dataset.N / dataset.batch_size)
        errors = list()
        for i in range(10):
            elem = sess.run(next_element['gaze'])
            def minmax(elem, ax):
                mi = np.min([e[ax] for e in elem])
                ma = np.max([e[ax] for e in elem])
                print(mi, ma)

            minmax(elem, 0)
            minmax(elem, 1)



            random_perm = elem.copy()
            random.shuffle(random_perm)

            input_gaze_unnormalised = elem * np.pi
            output_unnormalised = random_perm * np.pi
            new_errors = angular_error(input_gaze_unnormalised, output_unnormalised)

            errors.append(new_errors)
        print("Random mean: ", np.nanmean(errors))

            # print(elem['gaze'])
            # print(elem['gaze']/np.pi)
            # print(np.max(elem["gaze"], axis=1))
            # print(np.mean(np.max(np.abs(elem["gaze"]), axis=1)))
            # print(np.mean(np.mean(np.abs(elem["gaze"]), axis=1)))

            # from matplotlib.pyplot import imshow
            # from util.gaze import draw_gaze
            #
            # import matplotlib.pyplot as plt
            # for j in range(10):
            #     img = np.array((elem['eye'][j]+1) * 128,  dtype=np.int)
            #     gaze = elem['gaze'][j]
            #
            #     img = draw_gaze(
            #         img, (0.5 * img.shape[1], 0.5 * img.shape[0]),
            #         gaze, length=100.0, thickness=2, color=(0, 255, 0),
            #     )
            #     imshow(img)
            #     plt.title("Gaze: {:.3f} {:.3f}".format(*elem['gaze'][j]))
            #     plt.show()

