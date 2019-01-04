"""Translate an image to another image
An example of command-line usage is:
CUDA_VISIBLE_DEVICES=0 python run/fpgan_inference.py
    --config ../config/fpgan_basic.ini
    --section 20181123-1412
    --U2M True

"""
import json
import logging

import os
from collections import Iterable

import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.utils import convert2int
from util.config_loader import Config
from util.files import create_folder_if_not_exists
import numpy as np
from util.model_manager import ModelManager
from util.gaze import angular_error


class GazeNetInference:
    def __init__(self, path_in, saved_folder, batch_size, image_size, dataset_class):
        self.path_in = path_in
        # self.model_path = model_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset_class = dataset_class
        self.model_manager = ModelManager(saved_folder, [batch_size, *image_size, 3])

    def config_info(self):
        logging.info("Reading images from '{}'".format(self.path_in))
        logging.info("Batch size: {}".format(self.batch_size))

    @staticmethod
    def get_encoded_tensor(images):
        tensors_clean = convert2int(images)
        encoded_jpg = tf.map_fn(tf.image.encode_jpeg, tensors_clean,
                                dtype=tf.string)
        return encoded_jpg

    def run(self):
        self.config_info()
        counter = 0
        graph = tf.Graph()

        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                dm = DatasetManager()
                iterator = dm.get_dataset_iterator_for_path(
                    self.path_in,
                    image_size=self.image_size,
                    batch_size=self.batch_size,
                    shuffle=False,
                    repeat=False,
                    do_augmentation=False,
                    drop_remainder=True,
                    dataset_class=self.dataset_class
                )
                entry = iterator.get_next()
                input_tensor, output_tensor = self.model_manager.load_model(sess, graph)

                all_gaze_pred = {}
                all_gaze_true = {}

                errors_angular = list()
                errors_mse = list()

                coord = tf.train.Coordinator()
                while not coord.should_stop():
                    try:
                        logging.info("{} images processed".format(counter))
                        entry_evaluated = sess.run(entry)

                        gaze_pred = sess.run(
                            output_tensor,
                            { input_tensor: entry_evaluated['clean_eye']}
                        )

                        gaze_true = entry_evaluated['gaze']

                        for i, id in enumerate(entry_evaluated['id']):
                            id = id[0]
                            all_gaze_pred[id] = gaze_pred[i]
                            all_gaze_true[id] = gaze_true[i]

                        errors_angular += list(angular_error(gaze_true, gaze_pred))

                        counter += len(entry_evaluated['id'])
                        # if counter > 20:
                        #     raise KeyboardInterrupt()

                    except tf.errors.OutOfRangeError as e:
                        coord.request_stop()
                    except KeyboardInterrupt:
                        logging.info('Interrupted')
                        coord.request_stop()

                print("mean error (angular): ", np.mean(errors_angular))



def main():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('config', None, 'input configuration')
    tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')

    if FLAGS.config is None:
        print("Please provide config file (--config PATH).")
        exit()

    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)
    # Variables used for both directions
    batch_size = cfg.get('batch_size')
    image_size = [cfg.get('image_height'),
                  cfg.get('image_width')]

    path_in = cfg.get("path_test")
    checkpoint_folder = cfg.get('checkpoint_folder')
    assert checkpoint_folder is not None and checkpoint_folder != ""
    saved_folder = os.path.join(checkpoint_folder, "saved_model")
    dataset_class_test = cfg.get('dataset_class_test')
    inference = GazeNetInference(path_in, saved_folder, batch_size, image_size, dataset_class=dataset_class_test)
    inference.run()


if __name__ == '__main__':
    main()
