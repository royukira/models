from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = contrib_slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.' 
    'Select a sample to be masked with activation heatmap')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/roy_work/Classification/training',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'visual_dir', '/roy_work/Classification/visual', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'pci_HeadHat_dav4_cls', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/roy_work/Classification/data/pci_HeadHat_dav4_cls', 
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'mobilenet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 300, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool(
    'use_grayscale', False,
    'Whether to convert input images to grayscale.')

# Visual Set
tf.app.flags.DEFINE_string(
    'visual_scope', None,
    'Comma-separated list of scopes of variables to be visualized '
    'from a model.')

tf.app.flags.DEFINE_string(
    'visual_map_fn', 'p_power_sum',
    'The map function which maps a C*H*W of feature map tensor'
    'to a H*W activation-based heatmap; sum, p_power_sum, p_power_max'
)

tf.app.flags.DEFINE_integer(
    'p_power', 2,
    'The p power function'
)

FLAGS = tf.app.flags.FLAGS


# Visual map fn
def absolute_sum(feature_maps):
    """
    F(A) = sum(|a_i|), where i in the interval from 0 to num_channel of A
    
    Arg:
        -- feature_maps: A 4D tensor of [Batch_size, H, W, C] or a 3D tensor of [H, W, C]
    Return:
        A 3D tensor of [Batch_size, H, W] (not normalize the value between 0 and 1)
    """
    if len(feature_maps.shape) != 4: 
        if len(feature_maps.shape) == 3:
            feature_maps = tf.expand_dims(feature_maps, 0)
        else:
            raise ValueError("The shape:{} of feature maps tensor is illegal.".format(feature_maps.shape))
    
    return tf.reduce_sum(tf.abs(feature_maps),
                             axis=3)


def absolute_p_power_sum(feature_maps):
    """
    F(A) = sum(|a_i|^p), 
    where i in the interval from 0 to num_channel of A, p is the power
    
    Arg
        -- feature_maps: A 4D tensor of [Batch_size, H, W, C] or a 3D tensor of [H, W, C]
    Return:
        A 3D tensor of [Batch_size, H, W] (not normalize the value between 0 and 1)
    """
    if len(feature_maps.shape) != 4: 
        if len(feature_maps.shape) == 3:
            feature_maps = tf.expand_dims(feature_maps, 0)
        else:
            raise ValueError("The shape:{} of feature maps tensor is illegal.".format(feature_maps.shape))
    
    return tf.reduce_sum(tf.pow(tf.abs(feature_maps),
                                FLAGS.p_power),
                         axis=3
                        )


def absolute_p_power_max(feature_maps):
    """
    F(A) = max(|a_i|^p), 
    where i in the interval from 0 to num_channel of A, p is the power
    
    Arg:
        -- feature_maps: A 4D tensor of [Batch_size, H, W, C] or a 3D tensor of [H, W, C]
    Return:
        A 3D tensor of [Batch_size, H, W] (not normalize the value between 0 and 1)
    """
    if len(feature_maps.shape) != 4: 
        if len(feature_maps.shape) == 3:
            feature_maps = tf.expand_dims(feature_maps, 0)
        else:
            raise ValueError("The shape:{} of feature maps tensor is illegal.".format(feature_maps.shape))
    
    return tf.reduce_max(tf.pow(tf.abs(feature_maps),
                                FLAGS.p_power),
                         axis=3
                        )


# Visualize 
def _py_get_rgb_heatmap(image, label, predict, heatmap):
    """
    Mask an image with a heatmap in numpy way
    Note: This function need to be called by tf.py_func()

    Arg:
        -- image: a RGB image(distorted), type: np.array, 3D
        -- heatmap: a single-channel image, type: np.array, 2D
    Return:
        A masked image, type: np.array, 3D
    """
    if (len(image.shape) != 3) or (len(heatmap.shape) != 2):
        raise ValueError("The shape of image/heatmap is illegal.")
    
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  #  output is a 3-channel rgb image
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.putText(heatmap, "gt:{}".format(str(label)), (0, 30), 
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1,
                             (255, 255, 255),
                             2,
                             cv2.LINE_AA)
    heatmap = cv2.putText(heatmap, "predict:{}".format(str(predict)), (0, 60), 
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1,
                             (0, 0, 0),
                             2,
                             cv2.LINE_AA)

    return heatmap


def _mask_with_heatmap(image, label, predict, heatmap, scope_name, summaries=None):
    """
    Mask an image with a heatmap

    Arg:
        -- image: A 3D tensor of [H, W, 3]
        -- label: A tf.int64 tensor
        -- heatmap: a 2D tensor of [H, W]
        -- scope_name: the scope name of the variable
    Return:
        A masked image which is a 3D tensor of [H, W, 3]
    """
    if len(heatmap.shape) != 2:
        raise ValueError("The shape:{} of heatmap tensor is illegal.".format(heatmap.shape))

    with tf.name_scope('heatmap/{}'.format(scope_name)):
        if summaries is not None:
            summaries.add(tf.summary.image('rgb_image', tf.expand_dims(image, axis=0)))
        # Normalize the heatmap between 0 and 1
        norm_heatmap = heatmap / tf.reduce_max(heatmap)
        # The summary needs a 4D tensor [1,h,w,1]
        summary_norm_heatmap = tf.expand_dims(tf.expand_dims(tf.cast(norm_heatmap * 255, tf.uint8),
                                                    axis=2),
                                              axis=0
                                             )
        if summaries is not None:
            summaries.add(tf.summary.image('norm_heatmap', summary_norm_heatmap))
        # run py_func
        rgb_heatmap = tf.py_func(_py_get_rgb_heatmap, 
                                [image, label, predict, norm_heatmap], 
                                tf.uint8)

        if summaries is not None:
            summaries.add(tf.summary.image('rgb_heatmap', tf.expand_dims(rgb_heatmap, axis=0)))

        # mask
        intensity = 0.9
        norm_rgb_heatmap = tf.cast(rgb_heatmap, tf.float32) / tf.reduce_max(tf.cast(rgb_heatmap, tf.float32))
        masked_img = image + intensity * norm_rgb_heatmap
        #masked_img = tf.cast(masked_img, tf.uint8)

        # Add to summary
        masked_img = tf.expand_dims(masked_img, axis=0)
        if summaries is not None:
            summaries.add(tf.summary.image('masked_image', masked_img))
        return masked_img, summaries


def masked_with_heatmap(images, labels, predicts, heatmaps_dict, summarise=None, batch_index=[0]):
    """
    Mask the specific image with the correspondiing heatmap
    """
    if len(labels.shape) == 2:
        labels = tf.squeeze(labels)
    
    name_to_masked_img_list = []
    for idx in batch_index:
        image = images[batch_index]
        label = labels[batch_index]
        predict = predicts[batch_index]

        name_to_masked_img = {}
        for heatmap_name in heatmaps_dict.keys():
            heatmap = heatmaps_dict[heatmap_name][batch_index]
            masked_img, summarise = _mask_with_heatmap(image, label, predict, heatmap, heatmap_name, summarise)
            name_to_masked_img[heatmap_name] = masked_img
            name_to_masked_img_list.append(name_to_masked_img)
    return name_to_masked_img_list, summarise


VISUAL_MAP_FN = {
    'sum': absolute_sum,
    'p_power_sum': absolute_p_power_sum,
    'p_power_max': absolute_p_power_max,
}


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    if not FLAGS.visual_scope:
        raise ValueError('You must set the scopes of variables that you want to visualize')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False,
            use_grayscale=FLAGS.use_grayscale)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        
        # Pre-Processing a single image
        #  [ymin, xmin, ymax, xmax]
        box = tf.constant([0.2, 0.0, 0.75, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
        image = image_preprocessing_fn(image, eval_image_size, eval_image_size, bbox=box)
        
        # batch images
        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, end_points = network_fn(images)

        ###########################################################
        # extract the end_points which are going to be visualized #
        ###########################################################
        visual_map_fn = VISUAL_MAP_FN[FLAGS.visual_map_fn]  # TODO: debug those fn
        visual_endpoint_dict = {scope.strip():None
                            for scope in FLAGS.visual_scope.split(',')}
        heatmap_dict = {}
        for end_point_scope in end_points.keys():
            if end_point_scope in visual_endpoint_dict:
                # F(x) -> y, where x of B*C*H*W, y of B*H*W
                heatmap_dict[end_point_scope] = visual_map_fn(end_points[end_point_scope])
                visual_endpoint_dict[end_point_scope] = end_points[end_point_scope]
        
        if len(heatmap_dict.keys()) == 0:
            print(end_points.keys())
            print(visual_endpoint_dict)
            print(heatmap_dict)
            raise ValueError("No heatmap.")
        
        #print(end_points.keys())
        print(visual_endpoint_dict)
        print(heatmap_dict)
        input("Continue...")

        if FLAGS.quantize:
            contrib_quantize.create_eval_graph()

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)   # [batch_size, ]

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            summaries.add(op)

        # Mask the image with heatmaps from different layers & add to summary
        # Defaultly only visualize the first sample in a batch
        _, summaries = masked_with_heatmap(images,
                                           labels,
                                           predictions,
                                           heatmap_dict,
                                           summarise=summaries)

        # Add the masked images to summary
        # for scope_name, masked_img in scope_to_masked_img.items():
        #     summary_name = 'visual/{}'.format(scope_name)
        #     op = tf.summary.image(summary_name, masked_img)
        #     summaries.add(op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
        # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.visual_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            summary_op=summary_op,
            variables_to_restore=variables_to_restore)

if __name__ == "__main__":
    tf.app.run()