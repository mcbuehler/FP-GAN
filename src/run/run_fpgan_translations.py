"""Translate images from original to refined
An example of command-line usage is:
CUDA_VISIBLE_DEVICES=0 python run/run_fpgan_translations.py
    --config ../config/fpgan_basic.ini
    --section 20181123-1412
    --direction M2U

"""
import logging

import tensorflow as tf

from models.fpgan_export import GeneratorExport
from models.fpgan_inference import GeneratorInference
from util.config_loader import Config
from util.enum_classes import TranslationDirection as Direction, \
    DatasetClass as DS


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
        ngf=cfg.get("ngf"),
        rgb=cfg.get("rgb")
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
    rgb = cfg.get('rgb')

    shared_args = dict(
        batch_size=batch_size, image_size=image_size, rgb=rgb
    )
    # Variables dependent on direction
    if U2M:
        path_in = cfg.get("S")
        model_path = cfg.get("path_model_u2m")
        output_folder = cfg.get('path_refined_u2m')
        inference = GeneratorInference(path_in=path_in, model_path=model_path, output_folder=output_folder,
                                       dataset_class=DS.UNITY, **shared_args)
        inference.run()
    else:
        path_in = cfg.get("R")
        model_path = cfg.get("path_model_m2u")
        output_folder = cfg.get('path_refined_m2u')
        inference = GeneratorInference(path_in=path_in, model_path=model_path, output_folder=output_folder,
                                       dataset_class=DS.MPII, **shared_args)
        inference.run()


def main():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('config', None, 'input configuration')
    tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')
    tf.flags.DEFINE_string('direction', 'U2M', 'input configuration')

    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)

    if FLAGS.direction in [Direction.U2M, Direction.BOTH]:
        run_export(cfg, U2M=True)
        run_inference(cfg, U2M=True)
    if FLAGS.direction in [Direction.M2U, Direction.BOTH]:
        run_export(cfg, U2M=False)
        run_inference(cfg, U2M=False)
    if FLAGS.direction not in Direction.get_all():
        print("Invalid direction: {}".format(FLAGS.direction))
        print("Direction must be one of: {}".format(", ".join(Direction().get_all())))
        exit()




if __name__ == '__main__':
    main()