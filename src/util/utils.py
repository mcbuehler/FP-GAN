import numpy as np
import tensorflow as tf
import random


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
    """
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def batch_convert2int(images):
    """
    Args:
      images: 4D float tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D int tensor
    """
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def batch_convert2float(images):
    """
    Args:
      images: 4D int tensor (batch_size, image_size, image_size, depth)
    Returns:
      4D float tensor
    """
    return tf.map_fn(convert2float, images, dtype=tf.float32)


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = None

    def _init_images(self, images):
        self.images = images

    def _get_current_pool_size(self):
        if self.images is None:
            return 0
        else:
            return self.images.shape[0]

    def _add(self, images):
        if self._get_current_pool_size() == 0:
            self._init_images(images)
        else:
            self.images = np.concatenate((self.images, images), axis=0)

    def _refresh_pool(self, images, n):
        replace_indices_pool = self._sample_random_indices(self.images, n)
        replace_indices_new = self._sample_random_indices(images, n)
        self.images[replace_indices_pool] = images[replace_indices_new]

    def _sample_random_indices(self, array, n):
        return np.random.choice(range(len(array)), size=n, replace=False)

    def sample_from_pool(self, n):
        idx = self._sample_random_indices(self.images, n)
        return self.images[idx]

    def sample_from_images(self, images, n):
        idx = self._sample_random_indices(images, n)
        return images[idx]

    def query(self, images):
        """

        :param images: 4D image tensor where first dimension is batch size
        :return:
        """
        batch_size = images.shape[0]

        if self.pool_size == 0:
            # Why should we ever have this case?
            return images

        if self._get_current_pool_size() < self.pool_size:
            # Pool is not full yet. Add all current images
            # and return samples from entire pool.
            self._add(images)
            return self.sample_from_pool(batch_size)
        else:
            # Pool is full. We sample half of the entries from pool
            # and the other half from the new images
            batch_size_half = int(batch_size/2)
            sampled_pool_images = self.sample_from_pool(batch_size_half)
            sampled_new_images = self.sample_from_images(images, batch_size - batch_size_half)

            # We randomly replace half of the images in the pool with new images
            self._refresh_pool(images, batch_size_half)
            return np.concatenate((sampled_new_images, sampled_pool_images), axis=0)

