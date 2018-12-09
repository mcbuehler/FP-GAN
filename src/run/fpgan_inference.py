"""Translate an image to another image
An example of command-line usage is:
python fpgan_inference.py --input ../data/UnityEyesTest/
        --model ../checkpoints/20181123-1412/Unity2MPII.pb
"""
import logging

import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.files import save_images, create_folder_if_not_exists, copy_json
from util.utils import convert2int
from util.config_loader import Config

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')

if FLAGS.config is None:
    print("Please provide config file (--config PATH).")
    exit()


def config_info(path_in, model_path, output_folder, batch_size):
    logging.info("Reading images from '{}'".format(path_in))
    logging.info("Loading model from '{}'".format(model_path))
    logging.info("Writing images and json files to '{}'".format(output_folder))
    logging.info("Batch size: {}".format(batch_size))


def get_encoded_tensor(images):
    tensors_clean = convert2int(images)
    encoded_jpg = tf.map_fn(tf.image.encode_jpeg, tensors_clean,
                            dtype=tf.string)
    return encoded_jpg


def load_model(path_model, graph):
    with tf.gfile.FastGFile(path_model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())

        tf.import_graph_def(graph_def, name='output')

        input_tensor = graph.get_tensor_by_name('output/input_image:0')
        output_tensor = graph.get_tensor_by_name('output/output_image:0')
    return input_tensor, output_tensor


def inference(path_model, path_in, image_size, batch_size, output_folder):
    graph = tf.Graph()

    with graph.as_default():

        input_tensor, output_tensor = load_model(path_model, graph)

        with tf.Session(graph=graph) as sess:
            dm = DatasetManager()
            iterator = dm.get_dataset_iterator_for_path(
                path_in,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False,
                repeat=False,
                testing=True,
                drop_remainder=True
            )

            coord = tf.train.Coordinator()

            counter = 0

            while not coord.should_stop():
                try:
                    entry = iterator.get_next()
                    eyes_clean = entry['clean_eye']
                    ids = entry['id']

                    batch_ids, batch_eyes_clean = sess.run([ids, eyes_clean])

                    # Ids are returned as byte
                    image_ids = [id.decode('utf-8') for id in batch_ids]

                    generated = sess.run(output_tensor, feed_dict={
                                    input_tensor: batch_eyes_clean
                                })

                    save_images(generated, output_folder, image_ids)

                    # get clean images

                    images_clean = sess.run(get_encoded_tensor(batch_eyes_clean))
                    save_images(images_clean, output_folder, image_ids, suffix="_clean")

                    copy_json(image_ids, path_in, output_folder)
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
    batch_size = cfg.get('batch_size')
    model_path = cfg.get("path_model_u2m")
    image_size = [cfg.get('image_height'),
                  cfg.get('image_width')]
    output_folder = cfg.get('path_refined_u2m')
    path_in = cfg.get("S")
    # Info for the user
    config_info(path_in, model_path, output_folder, batch_size)

    create_folder_if_not_exists(output_folder)
    # Run inference
    inference(model_path, path_in, image_size, batch_size, output_folder)


if __name__ == '__main__':
    tf.app.run()
