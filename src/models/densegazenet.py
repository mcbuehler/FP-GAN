import tensorflow as tf

import util.ops as ops
import util.gaze as gaze
import util.utils as utils
from models.base_gazenet import BaseGazeNet
from util.ops_densenet import add_transition, add_layer, conv, global_average_pooling
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
-> l2 normalisation
-> l2 loss
LR = =.001
batch_size=512
"""


class DenseGazeNet(BaseGazeNet):
    # def __init__(self,
    #              name,
    #              batch_size=1,
    #              image_size=(78, 120),
    #              use_sigmoid=False,
    #              norm='instance',
    #              learning_rate=2e-4,
    #              beta1=0.9,
    #              beta2=0.999,
    #              tf_session=None
    #              ):
    #     """
    #     Args:
    #       name: name
    #
    #       batch_size: integer, batch size
    #       image_size: list(height, width)
    #       norm: 'instance' or 'batch'
    #       learning_rate: float, initial learning rate for Adam
    #       beta1: float, momentum term of Adam
    #       ngf: number of gen filters in first conv layer
    #       tf_session: tensorflow session. Needed for loading data sources.
    #     """
    #     super().__init__(name, batch_size=batch_size,
    #                      image_size=image_size,
    #                      use_sigmoid=use_sigmoid,
    #                      norm=norm,
    #                      learning_rate=learning_rate,
    #                      beta1=beta1,
    #                      beta2=beta2,
    #                      tf_session=tf_session
    #                      )

    def forward(self, input, is_training=True):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        self.is_training = tf.placeholder_with_default(is_training, shape=[],
                                                       name='is_training')
        # depth of network. This parameter will define how many layers will be in a dense block
        depth = 40
        # Number of layers per dense block
        N = int((depth - 4) / 3)
        # Number of feature maps added by a single convolutional layer
        growth_rate = 12

        # Test predictions only work if this is always true.
        # It is ok to set this to True. See https://piazza.com/class/jdbpmonr7fa26b?cid=105
        is_training = True

        with tf.variable_scope(self.name, reuse=self.reuse):
            # Initial convolution
            with tf.variable_scope('block_initial'):
                l = conv('conv0', input, 16, 1)

            # Three dense blocks
            with tf.variable_scope('block1'):
                for i in range(N):
                    l = add_layer('dense_layer.{}'.format(i), l, growth_rate=growth_rate, is_training=is_training)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2'):

                for i in range(N):
                    l = add_layer('dense_layer.{}'.format(i), l, growth_rate=growth_rate, is_training=is_training)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3'):

                for i in range(N):
                    l = add_layer('dense_layer.{}'.format(i), l, growth_rate=growth_rate, is_training=is_training)

            # Difference to original DenseNet: we output a numerical value
            with tf.variable_scope('regression'):
                l = tf.layers.batch_normalization(l, name='bnlast',
                                                  training=is_training)
                l = tf.nn.relu(l)
                l = global_average_pooling(name='gap', x=l)
                regressed_output = tf.layers.dense(l, units=2, name='fc4',
                                                   activation=None)
        # What about a layer that adds a restriction on output?
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.name)

        return regressed_output





