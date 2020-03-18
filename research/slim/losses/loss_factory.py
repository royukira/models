from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


def softmax_focal_loss(logits, 
                       onehot_labels, 
                       alpha=0.75, 
                       gamma=2.0, 
                       weights=1.0, 
                       label_smoothing=0, 
                       scope=None):
    """
    FL(p, y) = -alpha * pow(1-p, gamma) * log(p), if y = 1
    FL(p, y) = -(1-alpha) * pow(p, gamma) * log(1-p), otherwise 

    rewrite to

    FL(p) = alpha_factor * gamma_pow_factor * standard_xentropy
    
    The data we used which the classes are mutually exclusive 
    (each entry is in exactly one class), so we need to use softmax

    p = softmax(logits)
    """
    with tf.name_scope(scope, "focal_loss", 
                        [logits, onehot_labels, weights]) as scope:
        logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
        # `logits` and `labels` must have the same dtype
        onehot_labels = tf.cast(onehot_labels, logits.dtype)

        if label_smoothing > 0:
            num_classes = tf.cast(
                tf.shape(onehot_labels)[1], logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives
        
        """
        The data we used which the classes are mutually exclusive 
        (each entry is in exactly one class), so we need to use softmax 
        """
        # standard xentropy loss
        per_entry_xentropy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=onehot_labels, logits=logits, name="softmax_xentropy")
        
        # Calculate probabilities: y*p + (1-y) * (1-p)
        prediction_probabilities = tf.nn.softmax(logits)
        p_t = ((onehot_labels * prediction_probabilities) + 
                (1 - onehot_labels) * (1 - prediction_probabilities))
        
        # Calculate alpha part: y*alpha + (1-y) * (1-alpha)
        alpha_factor = ((onehot_labels * alpha) + 
                        (1 - onehot_labels) * (1 - alpha))
        
        # Calculate gamma power: pow(1 - (y*p + (1-y) * (1-p)), gamma)
        gamma_power_factor = tf.pow(1-p_t, gamma)

        # Calculate Focal loss: FL(p) = alpha_factor * gamma_pow_factor * standard_xentropy
        xentropy_focal_loss = alpha_factor * gamma_power_factor * per_entry_xentropy_loss

        # Calculate weight focal loss
        if weights:
            return slim.loss_ops.compute_weighted_loss(
                xentropy_focal_loss, weights=weights
            )
        else:
            return xentropy_focal_loss


def sigmoid_focal_loss():
    # TODO
    pass


def get_loss_fn(loss_fn_name):
    loss_fn_map = {
    'softmax_ce': slim.losses.softmax_cross_entropy,
    'softmax_focal_loss': softmax_focal_loss,
    'sigmoid_focal_loss': sigmoid_focal_loss
    }
    if (loss_fn_name in loss_fn_map):
        return loss_fn_map[loss_fn_name]
    else:
        raise ValueError('Name of loss function unknown %s' % loss_fn_name)