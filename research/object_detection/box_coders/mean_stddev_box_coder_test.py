<<<<<<< HEAD
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.box_coder.mean_stddev_boxcoder."""

import tensorflow as tf

from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list


class MeanStddevBoxCoderTest(tf.test.TestCase):

  def testGetCorrectRelativeCodesAfterEncoding(self):
    box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    boxes = box_list.BoxList(tf.constant(box_corners))
    expected_rel_codes = [[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]]
    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
    priors = box_list.BoxList(prior_means)

    coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
    rel_codes = coder.encode(boxes, priors)
    with self.test_session() as sess:
      rel_codes_out = sess.run(rel_codes)
      self.assertAllClose(rel_codes_out, expected_rel_codes)

  def testGetCorrectBoxesAfterDecoding(self):
    rel_codes = tf.constant([[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]])
    expected_box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    prior_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]])
    priors = box_list.BoxList(prior_means)

    coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
    decoded_boxes = coder.decode(rel_codes, priors)
    decoded_box_corners = decoded_boxes.get()
    with self.test_session() as sess:
      decoded_out = sess.run(decoded_box_corners)
      self.assertAllClose(decoded_out, expected_box_corners)


if __name__ == '__main__':
  tf.test.main()
=======
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.box_coder.mean_stddev_boxcoder."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.box_coders import mean_stddev_box_coder
from object_detection.core import box_list
from object_detection.utils import test_case


class MeanStddevBoxCoderTest(test_case.TestCase):

  def testGetCorrectRelativeCodesAfterEncoding(self):
    boxes = np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]], np.float32)
    anchors = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]], np.float32)
    expected_rel_codes = [[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]]

    def graph_fn(boxes, anchors):
      anchors = box_list.BoxList(anchors)
      boxes = box_list.BoxList(boxes)
      coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      rel_codes = coder.encode(boxes, anchors)
      return rel_codes

    rel_codes_out = self.execute(graph_fn, [boxes, anchors])
    self.assertAllClose(rel_codes_out, expected_rel_codes, rtol=1e-04,
                        atol=1e-04)

  def testGetCorrectBoxesAfterDecoding(self):
    rel_codes = np.array([[0.0, 0.0, 0.0, 0.0], [-5.0, -5.0, -5.0, -3.0]],
                         np.float32)
    expected_box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]
    anchors = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8]], np.float32)

    def graph_fn(rel_codes, anchors):
      anchors = box_list.BoxList(anchors)
      coder = mean_stddev_box_coder.MeanStddevBoxCoder(stddev=0.1)
      decoded_boxes = coder.decode(rel_codes, anchors).get()
      return decoded_boxes

    decoded_boxes_out = self.execute(graph_fn, [rel_codes, anchors])
    self.assertAllClose(decoded_boxes_out, expected_box_corners, rtol=1e-04,
                        atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
>>>>>>> master
