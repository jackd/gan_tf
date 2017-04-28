"""
Provides base class for generative-adversarial-network.

Interface designed to be similar to tf.estimator.Estimator class.

Implementation based on tf.estimator.Estimator class.
"""
import os
import inspect

import tensorflow as tf
from tensorflow.python.estimator import run_config
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training

from tensorflow.python.framework import ops


def _get_arguments(func):
    """Returns a spec of given func."""
    if hasattr(func, '__code__'):
        # Regular function.
        return inspect.getargspec(func)
    elif hasattr(func, '__call__'):
        # Callable object.
        return _get_arguments(func.__call__)
    elif hasattr(func, 'func'):
        # Partial function.
        return _get_arguments(func.func)


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = training.NewCheckpointReader(
            training.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(ops.GraphKeys.GLOBAL_STEP)
    except Exception:
        return 0


class Gan(object):
    """
    Base class for Generative Adversarial Networks.
    """

    def __init__(self, generator_fn, critic_logits_fn, model_dir=None,
                 config=None, params={}):
        """
        Construct a `Gan` instance.

        Args:
            generator_fn: Generator model function. Follows the signature:
                * Args:
                    * `features`: single `Tensor` or `dict` of `Tensor`s,
                        (depending on data passed to `train`)
                    * `mode`: optional. one of tf.estimator.ModeKeys
                    * `params`: optional `dict` of hyperparameters passed into
                        this function
                    * `config`: optional configuration opject passed into this
                        function, or the default config
                * Returns:
                    * generated_sample

            critic_logits_fn: Critic model function. Follows the signature:
                * Args:
                    * `features`: single `Tensor` or `dict` of `Tensor`s,
                        (depending on data passed to `train`/generator_fn
                        output)
                    * `params`: same as `generator_fn`
                    * `config`: same as `generator_fn`
                * Returns:
                    * critic_logits
        """
        if generator_fn is None or critic_logits_fn is None:
            raise ValueError(
                'generator_fn and critic_logits_fn cannot be None')

        if config is None:
            config = run_config.RunConfig()
        elif not isinstance(config, run_config.RunConfig):
            raise ValueError(
                'config must be an instance of RunConfig, but provided %s.' %
                config)

        self._config = config
        self._model_dir = model_dir
        self._params = params
        self._generator_fn = generator_fn
        self._critic_logits_fn = critic_logits_fn
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self._generator_kwargs = self._fn_kwargs(generator_fn)
        self._critic_logits_kwargs = self._fn_kwargs(critic_logits_fn)

    @property
    def model_dir(self):
        """Get the model directory for this GAN."""
        return self._model_dir

    def _fn_kwargs(self, fn):
        fn_args = _get_arguments(fn).args
        kwargs = {}
        if 'mode' in fn_args:
            kwargs['mode'] = None
        if 'params' in fn_args:
            kwargs['params'] = self._params
        if 'config' in fn_args:
            kwargs['config'] = self._config
        return kwargs

    def _call_generator_fn(self, features, mode):
        if mode in self._generator_kwargs:
            self._generator_kwargs['mode'] = mode
        return self._generator_fn(features=features, **self._generator_kwargs)

    def _call_critit_logits_fn(self, sample, mode):
        if mode in self._critic_logits_kwargs:
            self._critic_logits_kwargs['mode'] = mode
        return self._critic_logits_fn(sample, **self._critic_logits_kwargs)

    def get_scoped_generator_sample(
            self, features, mode=tf.estimator.ModeKeys.PREDICT, reuse=None):
        """Get the discriminator sample wrapped in a variable scope."""
        with tf.variable_scope('generator', reuse=reuse):
            sample = self._call_generator_fn(features, mode)
        return sample

    def get_scoped_critic_logits(
            self, sample, mode=tf.estimator.ModeKeys.PREDICT, reuse=None):
        """Get the critic logits wrapped in a variable scope."""
        with tf.variable_scope('critic', reuse=reuse):
            logits = self._call_critit_logits_fn(sample, mode)
        return logits

    def get_losses(self, real_logits, fake_logits):
        """
        Get losses associated with this GAN.

        Inputs:
            real_logits: logits associated with real inputs
            fake_logits: logits associated with fake inputs
        Returns:
            (c_loss, g_loss)
            c_loss: critic loss
            g_loss: generator loss
        """
        c_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits, labels=tf.ones_like(real_logits)))
        c_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        c_loss = c_real_loss + c_fake_loss
        g_loss = -c_fake_loss
        return c_loss, g_loss

    def critic_vars(self):
        """Get all variables associated with the discriminator."""
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    def generator_vars(self, graph=None):
        """Get all variables associated with the generator."""
        if graph is None:
            graph = tf
        return graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    def get_train_ops(self, generator_inputs, real_samples, global_step):
        """
        Get operations for training critic and generator.

        Returns:
            c_ops: critic train ops. Must contain at least loss and opt
            g_opt: generator train ops. Must contain at least loss and opt.
        """
        mode = tf.estimator.ModeKeys.TRAIN
        fake_samples = self.get_scoped_generator_sample(
            generator_inputs, mode)
        fake_logits = self.get_scoped_critic_logits(fake_samples, mode)
        real_logits = self.get_scoped_critic_logits(
            real_samples, mode, reuse=True)
        c_loss, g_loss = self.get_losses(real_logits, fake_logits)

        critic_vars = self.critic_vars()
        generator_vars = self.generator_vars()
        c_lr = self._params['critic_learning_rate'] \
            if 'critic_learning_rate' in self._params else 1e-4
        g_lr = self._params['generator_learning_rate'] \
            if 'generator_learning_rate' in self._params else 1e-4

        c_opt = tf.train.RMSPropOptimizer(c_lr).minimize(
            c_loss, var_list=critic_vars, global_step=global_step)
        g_opt = tf.train.RMSPropOptimizer(g_lr).minimize(
            g_loss, var_list=generator_vars, global_step=global_step)

        c_ops = {'loss': c_loss, 'opt': c_opt}
        g_ops = {'loss': g_loss, 'opt': g_opt}
        return c_ops, g_ops

    def train(self, generator_input_fn, real_sample_fn, real_iters_per_fake=5,
              steps=None, max_steps=None):
        """
        Train the adversarial network.

        Inputs:
            generator_input_fn: function that returns a tensor(s) which gives
                features for the generator input.
            real_generator_fn: function that returns a tensor(s) which
                generates real sample data.
        """
        if steps is not None and max_steps is not None:
            raise ValueError('Can not provide both steps and max_steps.')
        if max_steps is not None:
            start_step = _load_global_step_from_checkpoint_dir(self._model_dir)
            if max_steps <= start_step:
                logging.info(
                    'Skipping training since max_steps has already saved.')
                return self

        with ops.Graph().as_default() as g:
            global_step = training.create_global_step(g)
            generator_inputs = generator_input_fn()
            real_samples = real_sample_fn()

            c_ops, g_ops = self.get_train_ops(
                generator_inputs, real_samples, global_step)
            c_loss = c_ops['loss']
            g_loss = g_ops['loss']

            hooks = [
                basic_session_run_hooks.NanTensorHook(c_loss),
                basic_session_run_hooks.NanTensorHook(g_loss),
                basic_session_run_hooks.LoggingTensorHook(
                    {
                        'c_loss': c_loss,
                        'g_loss': g_loss,
                        'step': global_step,
                    },
                    every_n_iter=100),
                basic_session_run_hooks.StopAtStepHook(steps, max_steps)
            ]

            scaffold = monitored_session.Scaffold()

            chief_hooks = []
            if (self._config.save_checkpoints_secs or
                    self._config.save_checkpoints_steps):
                saver_hook_exists = any([
                  isinstance(h, training.CheckpointSaverHook)
                  for h in hooks
                ])
                if not saver_hook_exists:
                    chief_hooks = [
                        training.CheckpointSaverHook(
                            self._model_dir,
                            save_secs=self._config.save_checkpoints_secs,
                            save_steps=self._config.save_checkpoints_steps,
                            scaffold=scaffold)
                    ]

            with training.MonitoredTrainingSession(
                master=self._config.master,
                is_chief=self._config.is_chief,
                checkpoint_dir=self._model_dir,
                scaffold=scaffold,
                hooks=hooks,
                chief_only_hooks=chief_hooks,
                save_checkpoint_secs=0,  # saving handled by a hook?
                save_summaries_steps=self._config.save_summary_steps,
                config=config_pb2.ConfigProto(allow_soft_placement=True)
            ) as mon_sess:
                c_op_vals = {'loss': None}
                g_op_vals = {'loss': None}
                while not mon_sess.should_stop():
                    for i in range(real_iters_per_fake):
                        c_op_vals = mon_sess.run(c_ops)

                    g_op_vals = mon_sess.run(g_ops)
            logging.info('Last evaluated losses: critic: %s, generator, %s'
                         % (c_op_vals['loss'], g_op_vals['loss']))
        return self

    def _run_sess(self, outputs, iterate_batches):
        checkpoint_path = saver.latest_checkpoint(self._model_dir)

        with training.MonitoredSession(
            session_creator=monitored_session.ChiefSessionCreator(
                checkpoint_filename_with_path=checkpoint_path,
                config=config_pb2.ConfigProto(allow_soft_placement=True))
        ) as mon_sess:
            while not mon_sess.should_stop():
                output_vals = mon_sess.run(outputs)
                if iterate_batches:
                    yield output_vals
                else:
                    for output_val in output_vals:
                        yield output_val

    def generate(self, generator_input_fn, iterate_batches=True,
                 mode=tf.estimator.ModeKeys.PREDICT):
        """Generate data based on inputs generated by generator_input_fn."""
        generator_input = generator_input_fn()
        with tf.variable_scope('generator'):
            sample = self._call_generator_fn(
                generator_input, mode)
        return self._run_sess(sample, iterate_batches=iterate_batches)

    def discriminate(self, sample_input_fn, iterate_batches=True,
                     mode=tf.estimator.ModeKeys.PREDICT):
        """Get the model logits that the output of sample_input_fn is real."""
        sample = sample_input_fn()
        with tf.variable_scope('critic'):
            logits = self._call_critit_logits_fn(sample, mode)
            probs = tf.nn.sigmoid(logits)
        return self._run_sess(probs, iterate_batches=iterate_batches)
