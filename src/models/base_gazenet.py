import tensorflow as tf

import util.ops as ops
import util.gaze as gaze
import util.utils as utils

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


class BaseGazeNet:
    def __init__(self,
                 name,
                 batch_size=1,
                 image_size=(78, 120),
                 use_sigmoid=False,
                 norm='instance',
                 learning_rate=2e-4,
                 beta1=0.9,
                 beta2=0.999,
                 Optimiser=tf.train.AdamOptimizer,
                 tf_session=None
                 ):
        """
        Args:
          name: name

          batch_size: integer, batch size
          image_size: list(height, width)
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
          tf_session: tensorflow session. Needed for loading data sources.
        """

        self.name = name
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

        self.use_sigmoid = use_sigmoid
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.Optimiser = Optimiser
        self.tf_session = tf_session

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

    def get_loss(self, iterator, mode, is_training=True, regulariser=None):
        # # summary
        summary_pref = mode

        input_batch = iterator.get_next()

        input_eye = input_batch['eye']
        input_gaze = input_batch['gaze']
        output = self.forward(input_eye, mode=mode, is_training=is_training)
        error_angular = gaze.tensorflow_angular_error_from_pitchyaw(input_gaze, output)

        loss_gaze = tf.reduce_mean(tf.squared_difference(output, input_gaze))

        if regulariser is not None:
            # We add a regulariser
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regulariser,
                                                          reg_variables)
            tf.summary.scalar(self.create_name('loss/l2_reg', summary_pref),
                              reg_term)

            loss = loss_gaze + reg_term
        else:
            # we do not regularise
            loss = loss_gaze

        if is_training:
            # Create summaries
            tf.summary.image(self.create_name('input/eye', summary_pref), input_eye, max_outputs=1)

            tf.summary.histogram(self.create_name('input/eye', summary_pref), input_eye)
            tf.summary.histogram(self.create_name('input/gaze', summary_pref), input_gaze)
            tf.summary.histogram(self.create_name('output/gaze', summary_pref), output)

            tf.summary.scalar(self.create_name('loss/gaze_mse', summary_pref), loss_gaze)
            tf.summary.scalar(self.create_name('angular_error', summary_pref), error_angular)

        return {'gaze': output, 'error_angular': error_angular}, loss

    def optimize(self, loss):
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
            # tf.summary.scalar('learning_rate/{}'.format(name), self.learning_rate)

            learning_step = (
                # tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,
                #                                                                global_step,
                #                                                                variables)
                self.Optimiser(self.learning_rate, beta1=self.beta1, beta2=self.beta2, name=name)
                    .minimize(loss, global_step=global_step,
                              var_list=variables)
            )
            return learning_step

        optimiser = make_optimizer(loss, self.variables, name='Adam')

        with tf.control_dependencies(
                [optimiser]):
            return tf.no_op(name='optimisers')

