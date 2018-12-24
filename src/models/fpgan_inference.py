"""Translate an image to another image
An example of command-line usage is:
CUDA_VISIBLE_DEVICES=0 python run/fpgan_inference.py
    --config ../config/fpgan_basic.ini
    --section 20181123-1412
    --U2M True

"""
import logging

import os
import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.utils import convert2int
from util.config_loader import Config
from shutil import copyfile

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')
tf.flags.DEFINE_boolean('U2M', True, 'Direction of inference (M2U or U2M)')


if FLAGS.config is None:
    print("Please provide config file (--config PATH).")
    exit()


class GeneratorInference:
    def __init__(self, path_in, model_path, output_folder, batch_size, image_size):
        self.path_in = path_in
        self.model_path = model_path
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.image_size = image_size

    def config_info(self):
        logging.info("Reading images from '{}'".format(self.path_in))
        logging.info("Loading model from '{}'".format(self.model_path))
        logging.info("Writing images and json files to '{}'".format(self.output_folder))
        logging.info("Batch size: {}".format(self.batch_size))

    @staticmethod
    def get_encoded_tensor(images):
        tensors_clean = convert2int(images)
        encoded_jpg = tf.map_fn(tf.image.encode_jpeg, tensors_clean,
                                dtype=tf.string)
        return encoded_jpg

    def load_model(self, graph):
        with tf.gfile.FastGFile(self.model_path, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())

            tf.import_graph_def(graph_def, name='output')

            input_tensor = graph.get_tensor_by_name('output/input_image:0')
            output_tensor = graph.get_tensor_by_name('output/output_image:0')
        return input_tensor, output_tensor

    def save_images(self, tensor_generated, image_names, suffix=''):
        assert len(tensor_generated) == len(image_names)
        for i in range(len(tensor_generated)):
            filepath = os.path.join(self.output_folder,
                                    "{}{}.jpg".format(image_names[i], suffix))
            with open(filepath, 'wb') as f:
                f.write(tensor_generated[i])

    def copy_json(self, image_ids):
        for image_id in image_ids:
            filename = "{}.json".format(image_id)
            path_from = os.path.join(self.path_in, filename)
            path_to = os.path.join(self.output_folder, filename)
            copyfile(path_from, path_to)

    def run(self):
        self.config_info()

        graph = tf.Graph()

        with graph.as_default():

            input_tensor, output_tensor = self.load_model(graph)

            with tf.Session(graph=graph) as sess:
                dm = DatasetManager()
                iterator = dm.get_dataset_iterator_for_path(
                    self.path_in,
                    image_size=self.image_size,
                    batch_size=self.batch_size,
                    shuffle=False,
                    repeat=False,
                    testing=True,
                    drop_remainder=True,
                    dataset_class="unity"
                )

                coord = tf.train.Coordinator()

                counter = 0
                entry = iterator.get_next()

                while not coord.should_stop():
                    try:
                        eyes_clean = entry['clean_eye']
                        ids = entry['id']

                        batch_ids, batch_eyes_clean = sess.run([ids, eyes_clean])

                        # Ids are returned as byte
                        image_ids = [id.decode('utf-8') for id in batch_ids]

                        generated = sess.run(output_tensor, feed_dict={
                                        input_tensor: batch_eyes_clean
                                    })

                        self.save_images(generated, image_ids)

                        # get clean images
                        encoded_tensor = self.get_encoded_tensor(batch_eyes_clean)
                        images_clean = sess.run(encoded_tensor)
                        self.save_images(images_clean, image_ids, suffix="_clean")

                        self.copy_json(image_ids)
                        counter += len(image_ids)

                        logging.info("Processed {} images".format(counter))
                    except tf.errors.OutOfRangeError as e:
                        coord.request_stop()
                    except KeyboardInterrupt:
                        logging.info('Interrupted')
                        coord.request_stop()


def main(unused_argv):
    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)
    # Variables used for both directions
    batch_size = cfg.get('batch_size_inference')
    image_size = [cfg.get('image_height'),
                  cfg.get('image_width')]
    # Variables dependent on direction
    if FLAGS.U2M:
        path_in = cfg.get("S")
        model_path = cfg.get("path_model_u2m")
        output_folder = cfg.get('path_refined_u2m')
        inference = GeneratorInference(path_in, model_path, output_folder,
                                       batch_size, image_size)
        inference.run()
    else:
        logging.warning("Not implemented. Exiting.")
        exit(0)


if __name__ == '__main__':
    tf.app.run()
