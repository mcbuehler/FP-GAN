"""Translate an image to another image
An example of command-line usage is:
python fpgan_inference.py --input ../data/UnityEyesTest/
        --model ../checkpoints/20181123-1412/Unity2MPII.pb
"""
import logging
import os

import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.files import save_images, create_folder_if_not_exists, copy_json
from util.utils import convert2int


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 128, 'batch size, default: 512')
tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', '', 'input path (folder containing images)')

tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')

batch_size = FLAGS.batch_size

# Process input path
path_elements = FLAGS.input.split("/")
# If the input directory has a trailing slash we wouldn't extract the correct term
input_basename = path_elements[-1] if len(path_elements[-1]) else path_elements[-2]

# Process output path
image_size = [FLAGS.image_height, FLAGS.image_width]
model_path = os.path.dirname(FLAGS.model)
output_subfolder = "refined_{}_{}".format(
    input_basename,
    os.path.basename(FLAGS.model)[:-3])
output_folder = os.path.join(model_path, output_subfolder)
create_folder_if_not_exists(output_folder)

logging.info("Reading images from '{}'".format(FLAGS.input))
logging.info("Loading model from '{}'".format(model_path))
logging.info("Writing images and json files to '{}'".format(output_folder))


def get_encoded_tensor(images):
    tensors_clean = convert2int(images)
    encoded_jpg = tf.map_fn(tf.image.encode_jpeg, tensors_clean,
                            dtype=tf.string)
    return encoded_jpg



def inference():
    graph = tf.Graph()

    with graph.as_default():

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())

            tf.import_graph_def(graph_def, name='output')

            input_tensor = graph.get_tensor_by_name('output/input_image:0')
            output_tensor = graph.get_tensor_by_name('output/output_image:0')

        with tf.Session(graph=graph) as sess:
            dm = DatasetManager()
            iterator = dm.get_dataset_iterator_for_path(
                FLAGS.input,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False,
                repeat=False,
                testing=True
            )

            coord = tf.train.Coordinator()

            counter = 0

            while not coord.should_stop():
                try:
                    entry = iterator.get_next()
                    eyes = entry['eye']
                    eyes_clean = entry['clean_eye']
                    ids = entry['id']
                    batch_eyes, batch_ids, batch_eyes_clean = sess.run([eyes, ids, eyes_clean])
                    # Ids are returned as byte
                    image_ids = [id.decode('utf-8') for id in batch_ids]

                    generated = sess.run(output_tensor, feed_dict={
                                    input_tensor: batch_eyes
                                })
                    save_images(generated, output_folder, image_ids)

                    # get clean images

                    images_clean = sess.run(get_encoded_tensor(eyes_clean))
                    save_images(images_clean, output_folder, image_ids, suffix="_clean")

                    images_clean = sess.run(get_encoded_tensor(eyes))
                    save_images(images_clean, output_folder, image_ids, suffix="_original")


                    copy_json(image_ids, FLAGS.input, output_folder)
                    counter += len(image_ids)

                    logging.info("Processed {} images".format(counter))
                except tf.errors.OutOfRangeError as e:
                    coord.request_stop()
                except KeyboardInterrupt:
                    logging.info('Interrupted')
                    coord.request_stop()


def main(unused_argv):
    inference()


if __name__ == '__main__':
    tf.app.run()
