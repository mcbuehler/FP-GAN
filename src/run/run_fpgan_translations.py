"""Translate images from original to refined
An example of command-line usage is:
CUDA_VISIBLE_DEVICES=0 python run/run_fpgan_translations.py
    --config ../config/fpgan_basic.ini
    --section 20181123-1412
    --U2M True

"""
import logging

import tensorflow as tf

from models.fpgan_export import GeneratorExport
from models.fpgan_inference import GeneratorInference
from util.config_loader import Config

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')
tf.flags.DEFINE_boolean('U2M', True, 'Direction of inference (M2U or U2M)')


def run_export(cfg, U2M=True):
    """
    Exports the model in given direction (default U2M).
    :param cfg: config
    :param U2M: Export from Unity to MPII or the other way round
    :return:
    """
    logging.info("Exporting model from {}...".
                 format(cfg.get("checkpoint_folder")))
    default_args = dict(
        checkpoint_dir=cfg.get("checkpoint_folder"),
        image_size=[cfg.get('image_height'),
                  cfg.get('image_width')],
        batch_size=cfg.get("batch_size_inference"),
        norm=cfg.get("norm"),
        ngf=cfg.get("ngf")
    )

    generator_export = GeneratorExport(**default_args)

    if U2M:
        logging.info('Exporting U2M model...')
        generator_export.run(
            model_name=cfg.get("model_name_u2m"),
            U2M=True)
    else:
        logging.info('Exporting M2U model...')
        generator_export.run(
            model_name=cfg.get("model_name_m2u"),
            U2M=False)


def run_inference(cfg, U2M=True):
    """
    Translates images
    pre: there exists exported model using the same batch size
    :param cfg: config
    :param U2M: Unity to MPII
    :return:
    """
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


def main(unused_argv):
    # Load the config variables
    cfg_section = FLAGS.section
    cfg = Config(FLAGS.config, cfg_section)


    U2M = FLAGS.U2M
    run_export(cfg, U2M)
    run_inference(cfg, U2M)


if __name__ == '__main__':
    tf.app.run()