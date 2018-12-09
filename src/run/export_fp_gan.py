""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:

python run/export_fp_gan.py --config ../config/fpgan_basic.ini --section 20181207-1957
"""
from util.config_loader import Config
import tensorflow as tf

from models.model import CycleGAN


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')

if FLAGS.config is None:
    print("Please provide config file (--config PATH).")
    exit()


def export_graph(model_name, checkpoint_dir, image_size, batch_size, norm, ngf, U2M=True):
    graph = tf.Graph()
    input_dimensions = [batch_size, *image_size, 3]

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=ngf, norm=norm,
                             image_size=image_size)

        input_image = tf.placeholder(tf.float32, shape=input_dimensions,
                                     name='input_image')
        cycle_gan.model(fake_input=True)
        if U2M:
            output_image = cycle_gan.G.sample(input_image)
        else:
            output_image = cycle_gan.F.sample(input_image)

        output_image = tf.identity(output_image, name='output_image')
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        print("Latest checkpoint: {}".format(latest_ckpt))

        saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def, checkpoint_dir,
                             model_name, as_text=False)


def main(unused_argv):
    # Load the config variables
    cfg_section = FLAGS.section
    cfg = Config(FLAGS.config, cfg_section)

    print("Exporting model from {}...".format(cfg.get("checkpoint_folder")))
    print('Export U2M model...')
    default_args = dict(
        checkpoint_dir=cfg.get("checkpoint_folder"),
        image_size=[cfg.get('image_height'),
                  cfg.get('image_width')],
        batch_size=cfg.get("batch_size"),
        norm=cfg.get("norm"),
        ngf=cfg.get("ngf")
    )

    export_graph(
        cfg.get("model_name_u2m"),
        **default_args,
        U2M=True)
    print('Export M2U model...')
    export_graph(
        cfg.get("model_name_m2u"),
        **default_args,
        U2M=False)


if __name__ == '__main__':
    tf.app.run()
