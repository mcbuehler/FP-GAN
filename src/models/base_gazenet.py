import numpy as np
import tensorflow as tf
from tensorflow import GraphKeys

import util.gaze as gaze
from util.enum_classes import Mode

"""
Original architecture:
Input (35x55)
-> Conv3x3 (f=32)
-> Conv3x3 (f=32)
-> Conv3x3 (f=64)
-> MaxPool3x3 (stride=2)
-> Conv3x3 (f=80)
-> Conv3x3 (f=192)
-> Maxpool2x2 (stride=2)
-> FC9600
-> FC1000
-> FC3
-> l2 normalisation  // ommitted in our case
-> l2 loss

Hyper-parameters:
LR = =.001
batch_size=512
"""


class BaseGazeNet:
    """
    Base class for eye gaze estimation networks
    """

    def __init__(self,
                 name,
                 batch_size=1,
                 image_size=(78, 120),
                 norm='instance',
                 learning_rate=2e-4,
                 beta1=0.9,
                 beta2=0.999,
                 Optimiser=tf.train.AdamOptimizer,
                 normalise_gaze=False
                 ):
        """
        Args:
          name: name
          batch_size: integer, batch size
          image_size: list(height, width)
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          beta2: second moment decay rate for Adam
          Optimiser: optimiser for learning, e.g. tf.train.AdamOptimizer
          normalise_gaze: Whether the gaze has been normalised
            from [-pi, pi] to [0,1]
        """
        self.name = name
        self.norm = norm
        self.reuse = tf.AUTO_REUSE
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Optimiser = Optimiser
        self.normalise_gaze = normalise_gaze

    def forward(self, input, mode, is_training=True):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """
        raise NotImplementedError("Implement in subclass")

    def create_name(self, name, prefix):
        return "{}/{}".format(prefix, name)

    def get_loss(self, iterator, mode, is_training=True, regulariser=None,
                 summary_key=GraphKeys.SUMMARIES):
        input_batch = iterator.get_next()

        input_eye = input_batch['eye']
        input_gaze = input_batch['gaze']
        output = self.forward(input_eye, mode=mode, is_training=is_training)

        if self.normalise_gaze:
            # We have gaze in the range from [-1, 1]
            # Convert gaze back to range [-pi, pi]
            input_gaze_unnormalised = input_gaze * np.pi
            output_unnormalised = output * np.pi
        else:
            # There is nothing to do.
            input_gaze_unnormalised = input_gaze
            output_unnormalised = output

        # We apply standard MSE
        loss_gaze = tf.reduce_mean(tf.squared_difference(output_unnormalised,
                                                         input_gaze_unnormalised))

        if regulariser is not None:
            # We add a regulariser
            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regulariser,
                                                              reg_variables)
            tf.summary.scalar(self.create_name('loss/l2_reg', mode),
                              reg_term)

            loss = loss_gaze + reg_term
        else:
            # we do not regularise
            loss = loss_gaze

        # For easier interpretability we calculate the mean angular error
        error_angular = gaze.tensorflow_angular_error_from_pitchyaw(
            input_gaze_unnormalised, output_unnormalised)

        # Create summaries
        tf.summary.image(self.create_name('input/eye', mode), input_eye,
                         max_outputs=1, collections=[summary_key])

        tf.summary.histogram(self.create_name('input/eye', mode), input_eye,
                             collections=[summary_key])
        tf.summary.histogram(self.create_name('input/gaze', mode),
                             input_gaze_unnormalised,
                             collections=[summary_key])
        tf.summary.histogram(self.create_name('output/gaze', mode),
                             output_unnormalised, collections=[summary_key])

        tf.summary.scalar(self.create_name('loss/gaze_mse', mode), loss_gaze,
                          collections=[summary_key])
        tf.summary.scalar(self.create_name('angular_error', mode),
                          error_angular, collections=[summary_key])

        return {'gaze_output': output_unnormalised,
                'gaze_input': input_gaze_unnormalised,
                'error_angular': error_angular}, loss

    def optimize(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, trainable=False)
            learning_step = (
                self.Optimiser(self.learning_rate, beta1=self.beta1,
                               beta2=self.beta2, name="Adam")
                    .minimize(loss, global_step=global_step)
            )
        return learning_step

    def sample(self, input):
        """
        This is used for inference. Feed a placeholder or example image as
            input and this method will return the output.
        Args:
            input: sample input
        Returns: sample output
        """
        out = self.forward(input, mode=Mode.SAMPLE, is_training=False)
        return out
