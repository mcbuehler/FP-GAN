"""HDF5 data source for gaze estimation."""
from threading import Lock

import cv2 as cv
import h5py
import numpy as np
import tensorflow as tf
from typing import List, Tuple

from datasources.data_source import BaseDataSource


class HDF5Source(BaseDataSource):
    """HDF5 data loading class (using h5py)."""

    def __init__(self,
                 tensorflow_session: tf.Session,
                 batch_size: int,
                 hdf_path: str,
                 keys_to_use: List[str]=None,
                 eye_image_shape: Tuple[int, int]=None,
                 use_greyscale: bool=False,
                 num_reference_images=0,
                 testing: bool=False,
                 **kwargs):
        """Create queues and threads to read and preprocess data from specified keys."""
        self._eye_image_shape = eye_image_shape
        self._use_greyscale = use_greyscale
        self._num_reference_images = num_reference_images

        self._hdf_path = hdf_path
        self.hdf = h5py.File(hdf_path, 'r')
        self._short_name = 'HDF:%s' % '/'.join(hdf_path.replace('.h5', '').split('/')[-2:])
        if testing:
            self._short_name += ':test'

        # Record keys to use
        self._keys_to_use = keys_to_use
        if self._keys_to_use is None:
            self._keys_to_use = sorted([
                k for k in self.hdf.keys() if self.hdf[k]['gaze'].len() > 0
            ])
        assert len(self._keys_to_use) > 0
        first_key = self._keys_to_use[0]

        # Determine image key name
        self._image_key = None
        if self._image_key is None:
            self._image_key = 'eye' if 'eye' in self.hdf[first_key] else 'image'

        self._num_entries = np.sum([
            self.hdf[k][self._image_key].len() for k in self._keys_to_use
        ])
        self._num_people = len(self.hdf)

        self._mutex = Lock()
        self._current_index = 0
        self._current_key = None
        self._current_data = None
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

        # Set index to 0 again as base class constructor called HDF5Source::entry_generator once to
        # get preprocessed sample.
        self._current_index = 0
        self._current_key = None
        self._current_data = None

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._num_entries

    @property
    def short_name(self):
        """Short name specifying source HDF5."""
        return self._short_name

    def cleanup(self):
        """Close HDF5 file before running base class cleanup routine."""
        super().cleanup()

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0
            self._current_key = None
            self._current_data = None

    def entry_generator(self, yield_just_one=False, no_label=False):
        """Read entry from HDF5."""
        try:
            all_keys = sorted(list(self.hdf.keys()))

            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_data is None or \
                            self._current_index >= len(self._current_data[self._image_key]):
                        current_key_pos = (self._keys_to_use.index(self._current_key) \
                                           if self._current_key in self._keys_to_use else -1)
                        # If at last key, wrap around if training
                        if self._current_key == self._keys_to_use[-1]:
                            if self.testing:
                                break
                            current_key_pos = -1
                        self._current_key = self._keys_to_use[current_key_pos + 1]
                        self._current_index = 0

                        # Copy over person's data and shuffle
                        current_data = {}
                        if self.testing:
                            for name in (self._image_key, 'gaze', 'head'):
                                current_data[name] = np.copy(self.hdf[self._current_key+'/'+ name])
                        else:
                            num_to_sample_per_person = 200
                            num_person_entries = self.hdf[self._current_key+'/'+self._image_key].shape[0]
                            idxs = np.random.permutation(num_person_entries)
                            idxs = idxs[:min(num_to_sample_per_person, num_person_entries)]
                            idxs = sorted(idxs)
                            for name in (self._image_key, 'gaze', 'head'):
                                current_data[name] = np.copy(
                                    self.hdf[self._current_key+'/'+name][idxs, :],
                                )
                        self._current_data = current_data

                    # Retrieve data for indexing
                    current_key = self._current_key
                    current_index = self._current_index
                    current_data = self._current_data

                    # Increment index
                    self._current_index += 1

                # Create entry
                entry = {}
                for name in (self._image_key, 'gaze', 'head'):
                    entry[name] = current_data[name][current_index, :]

                # Sample reference stack of images
                if self._num_reference_images > 0:
                    num_entries = len(current_data[self._image_key])
                    ref_idxs = list(np.random.permutation(num_entries))
                    ref_idxs.remove(current_index)
                    ref_idxs = ref_idxs[:self._num_reference_images]
                    if len(ref_idxs) < self._num_reference_images:
                        continue
                    entry['references'] = current_data[self._image_key][ref_idxs, :]

                # Person identifier (should be unique in training set)
                # TODO: generalize across datasets?
                entry['person_id'] = np.array([all_keys.index(current_key)], dtype=np.int32)

                # BGR to RGB conversion
                entry["image"] = entry["image"][..., ::-1]

                if no_label:
                    yield entry['image']
                else:
                    yield entry
        finally:
            # Execute any cleanup operations as necessary
            pass

    def _preprocess_image(self, image):
        if self._eye_image_shape is not None:
            oh, ow = self._eye_image_shape
            image = cv.resize(image, (ow, oh))
        is_greyscale = len(image.shape) == 2
        if self._use_greyscale and not is_greyscale:
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            is_greyscale = True
        image = self.equalize(image)
        # NOTE: Now assuming image is histogram equalized for performance reasons.
        image = image.astype(np.float32)
        image *= 2.0 / 255.0
        image -= 1.0
        if is_greyscale:
            image = np.expand_dims(image, -1 if self.data_format == 'NHWC' else 0)
        elif self.data_format == 'NCHW':
            image = image.transpose((2, 0, 1))
        return image

    def preprocess_entry(self, entry):
        """Resize eye image and normalize intensities."""
        entry[self._image_key] = self._preprocess_image(entry[self._image_key])
        if 'references' in entry:
            entry['references'] = np.array([
                self._preprocess_image(image) for image in entry['references']
            ])
            if self.data_format == 'NHWC':
                entry['references'] = np.transpose(entry['references'][:, :, :, 0], [1, 2, 0])
            else:
                entry['references'] = entry['references'][:, 0, :, :]

        # Ensure some values in an entry are 4-byte floating point numbers
        for key in ['head', 'gaze']:
            entry[key] = entry[key].astype(np.float32)

        if not self.testing:
            if np.any(np.isnan(entry['head'])):
                print('omg nan head pose found!!')
                return None

        return entry
