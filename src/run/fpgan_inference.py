"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""
import os
import tensorflow as tf

from reader import UnityReader
from unityeyes import UnityEyes
from util import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', '', 'input path (folder containing images)')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')

tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')


def create_folder_if_not_exists(path):
  if not os.path.exists(path):
    os.mkdir(path)


def _parse_function(filename):
  image_data = tf.read_file(filename)
  input_image = tf.image.decode_jpeg(image_data, channels=3)
  input_image = tf.image.resize_images(input_image, size=image_size)
  input_image = utils.convert2float(input_image)

  return input_image


def save_images(tensor, path, image_names):
    assert tensor.size == len(image_names)
    for i in range(tensor.size):
        filepath = "{}.jpg".format(os.path.join(path, image_names[i]))
        with open(filepath, 'wb') as f:
            f.write(tensor[i])

    pass


batch_size = 2
LIMIT = 34

image_size = [FLAGS.image_height, FLAGS.image_width]
model_path = os.path.dirname(FLAGS.model)
output_subfolder = "generated_{}".format(os.path.basename(FLAGS.model)[:-3])
output_folder = os.path.join(model_path, output_subfolder)
create_folder_if_not_exists(output_folder)




# all_images_names = list(filter(lambda x: x.endswith('jpg'), os.listdir(FLAGS.input)))
# all_images_paths = [os.path.join(FLAGS.input, f) for f in all_images_names]
# if LIMIT > 0:
#     all_images_paths = all_images_paths[:LIMIT]
# print(len(all_images_paths))





def inference():
  graph = tf.Graph()

  with graph.as_default():
    # filenames = tf.constant(all_images_paths)
    #
    # dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # dataset = dataset.map(_parse_function).batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())

        tf.import_graph_def(graph_def, name='output')

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        input_tensor = graph.get_tensor_by_name('output/input_image:0')
        output_tensor = graph.get_tensor_by_name('output/output_image:0')


    with tf.Session(graph=graph) as sess:

        image_queue = UnityEyes(sess, batch_size, FLAGS.input,
                                testing=True, eye_image_shape=image_size,
                                data_format="NHWC", shuffle=False)
        image_queue.create_and_start_threads()


        counter = 0
        while True:
            print("Counter: ", counter)
            try:
                # batch = sess.run(iterator.get_next())
                print(image_queue.output_tensors.keys())
                batch = sess.run(image_queue.output_tensors['clean_eye'])
                image_ids = sess.run(image_queue.output_tensors['image_id'])

                image_ids = [id.decode('utf-8') for id in image_ids]

                if batch.shape[0] == batch_size:
                    generated = sess.run(output_tensor, feed_dict={
                        input_tensor: batch
                    })
                    save_images(generated, output_folder, image_ids)
                    counter += 1

                    if counter * batch_size > LIMIT:
                        print("processed {} images. stopping".format(counter * batch_size))
                        break
                else:
                    # we don't have a full batch any more
                    # TODO: fill with placeholders
                    break
            except tf.errors.OutOfRangeError:
                break

            # with open(output_path+"/test.jpg", 'wb') as f:
            #     f.write(generated)


def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
