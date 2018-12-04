""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:

python run/export_fp_gan.py --checkpoint_dir ../checkpoints/20181123-1412
"""

import tensorflow as tf

from models.model import CycleGAN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 128, 'batch size, default: 512')

tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_string('XtoY_model', 'Unity2MPII.pb',
                       'XtoY model name, default: apple2orange.pb')
tf.flags.DEFINE_string('YtoX_model', 'MPII2Unity.pb',
                       'YtoX model name, default: orange2apple.pb')

tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

batch_size = FLAGS.batch_size
image_size = (FLAGS.image_height, FLAGS.image_width)
input_dimensions = [batch_size, image_size[0], image_size[1], 3]


def export_graph(model_name, XtoY=True):
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=FLAGS.ngf, norm=FLAGS.norm,
                             image_size=image_size)

        input_image = tf.placeholder(tf.float32, shape=input_dimensions,
                                     name='input_image')
        cycle_gan.model(fake_input=True)
        if XtoY:
            output_image = cycle_gan.G.sample(input_image)
        else:
            output_image = cycle_gan.F.sample(input_image)

        output_image = tf.identity(output_image, name='output_image')
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        print("Latest checkpoint: {}".format(latest_ckpt))
        saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def, FLAGS.checkpoint_dir,
                             model_name, as_text=False)


def main(unused_argv):
    print("Exporting model from {}...".format(FLAGS.checkpoint_dir))
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()
