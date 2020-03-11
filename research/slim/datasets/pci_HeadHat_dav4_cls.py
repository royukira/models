
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from datasets import dataset_utils

slim = contrib_slim

_FILE_PATTERN = '%s.record'  # train.record or dev.record

_SPLITS_TO_SIZES = {
    'train': 23763,
    'dev': 1321,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'height': 'The height of the RGB image',
    'width': 'The width of the RGB image',
    'label': 'The label id of the image, integer between 0 and 3',
    'label_text': 'The text of the label.'
}

# 4 classes: 
#   (0) hatOnHead
#   (1) Head
#   (2) HatOnOthers
#   (3) hatOnHead_HatOnOthers
_NUM_CLASSES = 4

# If set to false, will not try to set label_to_names in dataset
# by reading them from labels.txt or github.
LOAD_READABLE_NAMES = True


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
        split_name: A train/test split name.
        dataset_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
        reader: The TensorFlow reader type.

    Returns:
        A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # 第一步
    # 将example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        # 'image/height': tf.io.FixedLenFeature(
        #     [1], tf.int64, default_value=0),
        # 'image/width': tf.io.FixedLenFeature(
        #     [1], tf.int64, default_value=0),
        'image/encoded': tf.io.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature(
            (), tf.string, default_value='png'),
        'image/object/class/label': tf.io.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        # 'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }

    # 第二步
    # 将反序列化的数据组装成更高级的格式。由slim完成
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded', format_key='image/format'),
        # 'height': slim.tfexample_decoder.Tensor('image/height'),
        # 'width': slim.tfexample_decoder.Tensor('image/width'),
        'label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        # 'label_text': slim.tfexample_decoder.Tensor('image/object/class/text'),
    }
    
    # 解码器，进行解码
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    print("Successfully decoded!!!!!!!!!!")
    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir, filename='head_hat_label_map.txt'):
        labels_to_names = dataset_utils.read_label_file(dataset_dir, filename='head_hat_label_map.txt')

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)  #字典形式，格式为：id:class_name
