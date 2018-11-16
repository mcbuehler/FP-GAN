"""
Make sure to use data_format = "NHWC" for now. Before allowing the use of "NCHW", we need to update the code for Generator and Discriminator.
"""
import tensorflow as tf

from datasources.unityeyes import UnityEyes
from datasources.hdf5 import HDF5Source


class UnityReader():
  def __init__(self, file_path, image_size=256,
    batch_size=1, name='', tf_session=None):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.name = name
    with tf.name_scope(self.name):
      self.image_queue = UnityEyes(tf_session, batch_size, file_path, testing=False, eye_image_shape=image_size, data_format="NHWC", shuffle=True)
      self.image_queue.create_and_start_threads()

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    images = self.image_queue.output_tensors['eye']
    return images


class MPIIGazeReader():
  def __init__(self, file_path, image_size, batch_size=1, name='', tf_session=None):
    """
    Args:
      file_path: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.name = name

    with tf.name_scope(self.name):
      self.image_queue = HDF5Source(tf_session, batch_size, file_path, testing=False, eye_image_shape=image_size, data_format="NHWC", shuffle=True)
      self.image_queue.create_and_start_threads()

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    images = self.image_queue.output_tensors['image']
    return images

