"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""
import os
import tensorflow as tf

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



batch_size = 1
image_size = [FLAGS.image_height, FLAGS.image_width]
model_path = os.path.dirname(FLAGS.model)
output_subfolder = "generated_{}".format(os.path.basename(FLAGS.model)[:-3])
output_filename = os.path.basename(FLAGS.input)

output_folder = os.path.join(model_path, output_subfolder)
create_folder_if_not_exists(output_folder)
output_path = os.path.join(output_folder, output_filename)

all_images = [os.path.join(FLAGS.input, f) for f in filter(lambda x: x.endswith('jpg'), os.listdir(FLAGS.input)[:10])]
print(all_images)

# A vector of filenames.


def inference():
  graph = tf.Graph()

  with graph.as_default():
    filenames = tf.constant(all_images)

    dataset = tf.data.Dataset.from_tensor_slices((filenames))
    dataset = dataset.map(_parse_function).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_image_batch = iterator.get_next()

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())

        print(next_image_batch)
        [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': next_image_batch},
                          return_elements=['output_image:0'],
                          name='output')
        print(output_image)

    with tf.Session(graph=graph) as sess:
        generated = output_image.eval()
        print(generated)
        with open(output_path+"/test.jpg", 'wb') as f:
            f.write(generated)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
