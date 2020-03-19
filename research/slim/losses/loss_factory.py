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
    print(logits.shape)
    print(onehot_labels.shape)
    input("continue")
    with tf.name_scope(scope, "softmax_focal_loss", 
                        [logits, onehot_labels, weights]) as scope:
        logits.get_shape().assert_is_compatible_with(onehot_labels.get_shape())
        # `logits` and `labels` must have the same dtype
        onehot_labels = tf.cast(onehot_labels, logits.dtype)
        
        # num_classes = tf.cast(
        #         tf.shape(onehot_labels)[1], tf.int32)
        # num_batch = tf.cast(
        #         tf.shape(onehot_labels)[0], tf.int32)
        if label_smoothing > 0:
            num_classes = tf.cast(
                tf.shape(onehot_labels)[1], logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            onehot_labels = onehot_labels * smooth_positives + smooth_negatives
        
        """
        Noted by Roy

        The data we used which the classes are mutually exclusive 
        (each entry is in exactly one class), so we need to use softmax to the logits

        Note: softmax和sigmoid的不同之处是：softmax(logits)每个元素加起来是等于1，而
        sigmoid(logits)并不保证这个，只会把logits每个元素从[-inf, inf]映射到[0.0, 1.0]，
        所以：
            1、sigmoid适合多标签任务，即独立且不互斥
            2、softmax不适合多标签任务，每个样本只能有一个label，即独立且互斥

        # standard softmax xentropy loss
            
          per_entry_xentropy_loss = tf.nn.softmax_cross_entropy_with_logits(
               labels=onehot_labels, logits=logits, name="softmax_xentropy")
          
          "tf.nn.softmax_cross_entropy_with_logits" is equal to 
          "-tf.reduce_sum(labels * tf.log(logits_scaled), 1)",
          where logits_scaled = tf.nn.softmax(logits)
          
          ==========================================================
          For example:
          labels = [[0.2, 0.3, 0.5],
                    [0.1, 0.6, 0.3]]
          logits = [[2, 0.5, 1],
                    [0.1, 1, 3]]
          result1 = tf.nn.softmax_cross_entropy_with_logits(
              labels=labels, logits=logits)
          result2 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)

          >>> result 1: [ 1.41436887  1.66425455]
          >>> result 2: [ 1.41436887  1.66425455]
          ==========================================================


          so, it will return a loss tensor of (batch_size,)

          if the label tensor is one-hot tensor(it means all elements are 0, except a
          positive label of each entry which is 1), element in each entry of the loss 
          tensor is equal to 
                               -y_i * log(logits_scaled_i) where i is arg(y_i = 1)   [1]
          (only calculating the losses of positive labels, negative labels are 0 so 
          the corresponding losses are 0)

          if the label tensor is smoothed (e.g [0, 0, 1] -> [0.05, 0.05, 0.9]), element in 
          each entry of the loss tensor is equal to 
                    -reduce_sum(y_i * log(logits_scaled_i*)) where i belong to [0, num_classes)  [2]

        # softmax focal loss
          In this function, the softmax focal loss is:
                    -alpha * pow(1-logits_scaled, gamma) * log(logits_scaled),  when y = 1    [3]
                    -(1-alpha) * pow(logits_scaled, gamma) * log(1 - logits_scaled), others   [4]
          rewrite to:
                    -reduce_sum(alpha_factor * gamma_factor * (y * log(logits_scaled)))   [5]
          where 
                    alpha_factor = y*alpha + (1-y) * (1-alpha)                            [6]
                    gamma_factor = pow(1 - (y*p + (1-y) * (1-p)), gamma)                  [7]

          Note: 
          the situations of the different label tensor type are similar to the above standard xentropy loss
        
        Note: 根据上面公式，所以实现softmax_focal_loss和sigmoid_focal_loss的不同之处在于，per_entry_xentropy_loss
        不能直接用softmax_cross_entropy_with_logits求得，因为里面自带了reduce_sum，用这个相当于变成了：
        
                    alpha_factor * gamma_factor * (-reduce_sum(y*log(logits_scaled)))
        
        如果是label tensor 是 one-hot, 那么这个其实是等于上面公式[5]的；但是如果label tensor不是one-hot，或者被soomthed
        那么这就和上面公式[5]不相等了
        """
        # Calculate probabilities per classes: y*p + (1-y) * (1-p)
        prediction_probabilities = tf.nn.softmax(logits)
        # p_t = ((onehot_labels * prediction_probabilities) + 
        #         (1 - onehot_labels) * (1 - prediction_probabilities))
        p_t = onehot_labels * prediction_probabilities

        # Calculate xentropy per entry (aka. per classes)
        # per_entry_xentropy = ((onehot_labels * tf.log(prediction_probabilities)) + 
        #         (1 - onehot_labels) * (1 - tf.log(prediction_probabilities)))
        per_entry_xentropy = onehot_labels * tf.log(prediction_probabilities)

        # Calculate alpha part: y*alpha + (1-y) * (1-alpha)
        # alpha_factor = ((onehot_labels * alpha) + 
        #                 (1 - onehot_labels) * (1 - alpha))
        alpha_factor = onehot_labels * alpha

        # Calculate gamma power: pow(1 - (y*p + (1-y) * (1-p)), gamma)
        gamma_power_factor = tf.pow(1.0 - p_t, gamma)

        # Calculate Focal loss: FL(p) = alpha_factor * gamma_pow_factor * standard_xentropy
        print(per_entry_xentropy.shape)
        print(alpha_factor.shape)
        print(gamma_power_factor.shape)
        input("continue")
        xentropy_focal_loss = -tf.reduce_sum(alpha_factor * gamma_power_factor * per_entry_xentropy,
                                             axis=1)
        print(xentropy_focal_loss.shape)
        input("continue")
        # Calculate weight focal loss
        if weights:
            return slim.loss_ops.compute_weighted_loss(
                xentropy_focal_loss, 
                weights=weights, 
                scope=scope
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