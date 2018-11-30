import tensorflow as tf

import util.ops as ops
import util.gaze as gaze
import util.utils as utils

"""
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


class GazeNet:
    def __init__(self,
                 name,
                 is_training,
                 batch_size=1,
                 image_size=(78, 120),
                 use_sigmoid=False,
                 norm='instance',
                 learning_rate=2e-4,
                 beta1=0.9,
                 beta2=0.999,
                 tf_session=None
                 ):
        """
        Args:
          X_train_file: string UnityEyes folder
          Y_train_file: string MPIIGaze h5 File
          batch_size: integer, batch size
          image_size: list(height, width)
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          lambda_identity: integer, weight for
            identity transformation loss (X -> Y). Same for both directions.
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
          tf_session: tensorflow session. Needed for loading data sources.
        """

        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

        self.use_sigmoid = use_sigmoid
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.tf_session = tf_session

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

        with tf.variable_scope(self.name):
            # convolution layers
            c32_1 = ops.conv3x3(input, k=32, stride=2, reuse=self.reuse, norm=None,
                                is_training=self.is_training, name="c32_1", summary=True)
            c32_2 = ops.conv3x3(c32_1, k=32, reuse=self.reuse, norm=self.norm,
                                is_training=self.is_training, name="c32_2")
            c64 = ops.conv3x3(c32_2, k=64, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name="c64")
            maxpool3x3 = ops.maxpool(c64, 3, name="maxpool3x3", stride=2,
                                     reuse=self.reuse)
            c80 = ops.conv3x3(maxpool3x3, k=80, reuse=self.reuse,
                              norm=self.norm,
                              is_training=self.is_training, name="c80")
            c192 = ops.conv3x3(c80, k=192, reuse=self.reuse, norm=self.norm,
                               is_training=self.is_training, name="c192")
            maxpool2x2 = ops.maxpool(c192, 2, name="maxpool2x2", stride=2,
                                     reuse=self.reuse)
            flattened = tf.contrib.layers.flatten(maxpool2x2)
            fc9600 = ops.dense(flattened, d=9600, name="fc9600",
                               reuse=self.reuse)
            fc1000 = ops.dense(fc9600, d=1000, name="fc1000", reuse=self.reuse)
            out = ops.last_dense(fc1000, name="out", reuse=self.reuse, use_sigmoid=self.use_sigmoid)
            out_normalised = tf.nn.l2_normalize(out, axis=1, name="l2_normalise")
        # What about a layer that adds a restriction on output?
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.name)

        return out_normalised

    def create_name(self, name, prefix):
        return "{}/{}".format(prefix, name)

    def get_loss(self, input, is_training=True):
        input_eye = input['eye']
        input_gaze = input['gaze']
        output = self.forward(input_eye, is_training=is_training)

        loss_mse = tf.reduce_mean(tf.squared_difference(output, input_gaze))

        error_angular = gaze.tensorflow_angular_error_from_pitchyaw(input_gaze,output)

        # # summary
        summary_pref = "train" if is_training else "test"

        tf.summary.histogram(self.create_name('input/eye', summary_pref), input_eye)
        tf.summary.histogram(self.create_name('input/gaze', summary_pref), input_gaze)
        tf.summary.histogram(self.create_name('output/gaze', summary_pref), output)

        tf.summary.scalar(self.create_name('loss/mse', summary_pref), loss_mse)
        tf.summary.scalar(self.create_name('angular_error', summary_pref), error_angular)

        tf.summary.image(self.create_name('input', summary_pref), utils.batch_convert2int(input_eye))

        return {'gaze': output}, loss_mse

    def optimize(self, loss, n_steps=200000):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k
            steps (~100 epochs)
                and a linearly decaying rate that goes to zero over
                the next 100k steps
            """
            # why can global step be 0 here? Does it automatically
            # increase after each optimisation step?
            global_step = tf.Variable(0, trainable=False)
            # starter_learning_rate = self.learning_rate
            # end_learning_rate = 0.0
            # start_decay_step = int(n_steps / 2)
            # decay_steps = n_steps - start_decay_step
            # learning_rate = (
            #     tf.where(
            #         tf.greater_equal(global_step, start_decay_step),
            #         tf.train.polynomial_decay(starter_learning_rate,
            #                                   global_step - start_decay_step,
            #                                   decay_steps, end_learning_rate,
            #                                   power=1.0),
            #         starter_learning_rate
            #     )
            #
            # )
            tf.summary.scalar('learning_rate/{}'.format(name), self.learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2, name=name)
                    .minimize(loss, global_step=global_step,
                              var_list=variables)
            )
            return learning_step

        optimiser = make_optimizer(loss, self.variables, name='Adam')

        with tf.control_dependencies(
                [optimiser]):
            return tf.no_op(name='optimisers')

