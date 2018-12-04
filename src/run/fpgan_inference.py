"""Translate an image to another image
An example of command-line usage is:
python fpgan_inference.py --input ../data/UnityEyesTest/
        --model ../checkpoints/20181123-1412/Unity2MPII.pb
"""
import os

import numpy as np
import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.enum_classes import Mode
from util import utils
from util.files import save_images, create_folder_if_not_exists

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', '', 'input path (folder containing images)')

tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')


def _parse_function(filename):
    image_data = tf.read_file(filename)
    input_image = tf.image.decode_jpeg(image_data, channels=3)
    input_image = tf.image.resize_images(input_image, size=image_size)
    input_image = utils.convert2float(input_image)

    return input_image




batch_size = 2
LIMIT = 34

image_size = [FLAGS.image_height, FLAGS.image_width]
model_path = os.path.dirname(FLAGS.model)
output_subfolder = "generated_{}".format(os.path.basename(FLAGS.model)[:-3])
output_folder = os.path.join(model_path, output_subfolder)
create_folder_if_not_exists(output_folder)


def inference():
    graph = tf.Graph()

    with graph.as_default():

        with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())

            tf.import_graph_def(graph_def, name='output')

            # print("All nodes in graph: ")
            # print([n.name for n in tf.get_default_graph().as_graph_def().node])

            input_tensor = graph.get_tensor_by_name('output/input_image:0')
            output_tensor = graph.get_tensor_by_name('output/output_image:0')

        with tf.Session(graph=graph) as sess:
            iterator = DatasetManager().get_dataset_iterator(
                mode=Mode.INFERENCE_UNITY_TO_MPII,
                image_size=image_size,
                batch_size=batch_size,
                shuffle=False,
                repeat=False
            )

            while True:
                try:
                    entry = iterator.get_next()
                    print(entry)
                    eyes = entry['eye']
                    # ids = entry['id']
                    # batch_eyes, batch_ids = sess.run([eyes, ids])
                    batch_eyes = sess.run(eyes)
                    print(np.array(batch_eyes).shape)

                    generated = sess.run(output_tensor, feed_dict={
                                    input_tensor: batch_eyes
                                })
                    print(generated)
                    image_ids = ['1', '2']
                    save_images(generated, output_folder, image_ids)
                            # print("Saved {} images".format(len(image_ids)))
                    exit()
                except tf.errors.OutOfRangeError as e:
                    print(e)
                    print("error")
                    break
            # number_of_batches = int(number_of_entries / batch_size)
            #
            # print("number of entries: {}".format(number_of_entries))
            # print("batch size: {}".format(batch_size))
            # # print("number of batches: {}".format(number_of_batches))
            #
            # for i in range(number_of_batches):
            #     print("Step {} / {}".format(i, number_of_batches - 1))
            #     try:
            #         eyes = image_queue.output_tensors['clean_eye']
            #         ids = image_queue.output_tensors['image_id']
            #
            #         # Why does it freeze here in the last iteration?
            #         # eyes and ids still have two samples in there...
            #         batch_eyes, batch_ids = sess.run([eyes, ids])
            #         image_ids = [id.decode('utf-8') for id in batch_ids]
            #
            #         print("Processing image ids:")
            #         print(image_ids)
            #
            #         generated = sess.run(output_tensor, feed_dict={
            #             input_tensor: batch_eyes
            #         })
            #         save_images(generated, output_folder, image_ids)
            #         print("Saved {} images".format(len(image_ids)))
            #
            #         if i * batch_size > LIMIT:
            #             print("processed {} images. stopping".format(
            #                 i * batch_size))
            #             break
            #
            #     except tf.errors.OutOfRangeError:
            #         print("out of range")
            #         break


def main(unused_argv):
    inference()


if __name__ == '__main__':
    tf.app.run()
