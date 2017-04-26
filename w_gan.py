"""Provides Wasserstein GAN class WGan."""
import tensorflow as tf
from gan_tf.gan import Gan


class WGan(Gan):
    """Wasserstein GAN."""

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
        c_fake_loss = tf.reduce_mean(fake_logits)
        g_loss = -c_fake_loss
        c_real_loss = -tf.reduce_mean(real_logits)
        c_loss = c_fake_loss + c_real_loss

        tf.summary.scalar('c_loss', c_loss)
        tf.summary.scalar('g_loss', g_loss)
        return c_loss, g_loss

    def get_train_ops(self, real_logits, fake_logits, global_step):
        """
        Get operations for training critic and generator.

        Returns:
            c_ops: critic train ops. Must contain at least loss and opt
            g_opt: generator train ops. Must contain at least loss and opt.
        """
        c_ops, g_ops = super(WGan, self).get_train_ops(
            real_logits, fake_logits, global_step)

        critic_vars = self.critic_vars()
        clip_val = self._params['max_critic_var'] \
            if 'max_critic_var' in self._params else 1e-2
        c_clip = [p.assign(tf.clip_by_value(p, -clip_val, clip_val))
                  for p in critic_vars]
        c_ops['clip'] = c_clip
        return c_ops, g_ops
