        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image2',
              groundtruth_boxes:
                  np.array([[50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[75., 76.], [float('nan'),
                                          float('nan')],
                             [float('nan'), float('nan')], [77., 78.]]]),
              detection_boxes:
                  np.array([[50., 50., 100., 100.]]),
              detection_scores:
                  np.array([.7]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], 1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], 1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsAndVisibilities(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))
    groundtruth_keypoint_visibilities = tf.placeholder(
        tf.float32, shape=(None, 4))
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key:
            image_id,
        input_data_fields.groundtruth_boxes:
            groundtruth_boxes,
        input_data_fields.groundtruth_classes:
            groundtruth_classes,
        input_data_fields.groundtruth_keypoints:
            groundtruth_keypoints,
        input_data_fields.groundtruth_keypoint_visibilities:
            groundtruth_keypoint_visibilities,
        detection_fields.detection_boxes:
            detection_boxes,
        detection_fields.detection_scores:
            detection_scores,
        detection_fields.detection_classes:
            detection_classes,
        detection_fields.detection_keypoints:
            detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              groundtruth_keypoint_visibilities:
                  np.array([[0, 0, 0, 2]]),
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[50., 60.], [1., 2.], [3., 4.], [170., 180.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], -1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], -1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsIsAnnotated(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))
    is_annotated = tf.placeholder(tf.bool, shape=())
    detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_keypoints = tf.placeholder(tf.float32, shape=(None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        'is_annotated': is_annotated,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints,
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[150., 160.], [float('nan'),
                                            float('nan')],
                             [float('nan'), float('nan')], [170., 180.]]]),
              is_annotated:
                  True,
              detection_boxes:
                  np.array([[100., 100., 200., 200.]]),
              detection_scores:
                  np.array([.8]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image2',
              groundtruth_boxes:
                  np.array([[50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1]),
              groundtruth_keypoints:
                  np.array([[[75., 76.], [float('nan'),
                                          float('nan')],
                             [float('nan'), float('nan')], [77., 78.]]]),
              is_annotated:
                  True,
              detection_boxes:
                  np.array([[50., 50., 100., 100.]]),
              detection_scores:
                  np.array([.7]),
              detection_classes:
                  np.array([1]),
              detection_keypoints:
                  np.array([[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]])
          })
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image3',
              groundtruth_boxes:
                  np.zeros((0, 4)),
              groundtruth_classes:
                  np.zeros((0)),
              groundtruth_keypoints:
                  np.zeros((0, 4, 2)),
              is_annotated:
                  False,  # Note that this image isn't annotated.
              detection_boxes:
                  np.array([[25., 25., 50., 50.], [25., 25., 70., 50.],
                            [25., 25., 80., 50.], [25., 25., 90., 50.]]),
              detection_scores:
                  np.array([0.6, 0.7, 0.8, 0.9]),
              detection_classes:
                  np.array([1, 2, 2, 3]),
              detection_keypoints:
                  np.array([[[0., 0.], [0., 0.], [0., 0.], [0., 0.]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], 1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], 1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)

  def testGetOneMAPWithMatchingKeypointsBatched(self):
    category_keypoint_dict = _get_category_keypoints_dict()
    coco_keypoint_evaluator = coco_evaluation.CocoKeypointEvaluator(
        category_id=1, category_keypoints=category_keypoint_dict['person'],
        class_text='person')
    batch_size = 2
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    groundtruth_keypoints = tf.placeholder(
        tf.float32, shape=(batch_size, None, 4, 2))
    detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_keypoints = tf.placeholder(
        tf.float32, shape=(batch_size, None, 4, 2))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_keypoints: groundtruth_keypoints,
        detection_fields.detection_boxes: detection_boxes,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_keypoints: detection_keypoints
    }

    eval_metric_ops = coco_keypoint_evaluator.get_estimator_eval_metric_ops(
        eval_dict)

    _, update_op = eval_metric_ops['Keypoints_Precision/mAP ByCategory/person']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image1', 'image2'],
              groundtruth_boxes:
                  np.array([[[100., 100., 200., 200.]], [[50., 50., 100.,
                                                          100.]]]),
              groundtruth_classes:
                  np.array([[1], [3]]),
              groundtruth_keypoints:
                  np.array([[[[150., 160.], [float('nan'),
                                             float('nan')],
                              [float('nan'), float('nan')], [170., 180.]]],
                            [[[75., 76.], [float('nan'),
                                           float('nan')],
                              [float('nan'), float('nan')], [77., 78.]]]]),
              detection_boxes:
                  np.array([[[100., 100., 200., 200.]], [[50., 50., 100.,
                                                          100.]]]),
              detection_scores:
                  np.array([[.8], [.7]]),
              detection_classes:
                  np.array([[1], [3]]),
              detection_keypoints:
                  np.array([[[[150., 160.], [1., 2.], [3., 4.], [170., 180.]]],
                            [[[75., 76.], [5., 6.], [7., 8.], [77., 78.]]]])
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['Keypoints_Precision/mAP ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.50IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP@.75IOU ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Precision/mAP (medium) ByCategory/person'], -1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@1 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@10 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(metrics['Keypoints_Recall/AR@100 ByCategory/person'],
                           1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (large) ByCategory/person'], 1.0)
    self.assertAlmostEqual(
        metrics['Keypoints_Recall/AR@100 (medium) ByCategory/person'], -1.0)
    self.assertFalse(coco_keypoint_evaluator._groundtruth_list)
    self.assertFalse(coco_keypoint_evaluator._detection_boxes_list)
    self.assertFalse(coco_keypoint_evaluator._image_ids)


class CocoMaskEvaluationTest(tf.test.TestCase):

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image1',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image1',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[100., 100., 200., 200.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
            np.pad(np.ones([1, 100, 100], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image2',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image2',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[50., 50., 100., 100.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
            np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_ground_truth_image_info(
        image_id='image3',
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes:
            np.array([[25., 25., 50., 50.]]),
            standard_fields.InputDataFields.groundtruth_classes: np.array([1]),
            standard_fields.InputDataFields.groundtruth_instance_masks:
            np.pad(np.ones([1, 25, 25], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    coco_evaluator.add_single_detected_image_info(
        image_id='image3',
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes:
            np.array([[25., 25., 50., 50.]]),
            standard_fields.DetectionResultFields.detection_scores:
            np.array([.8]),
            standard_fields.DetectionResultFields.detection_classes:
            np.array([1]),
            standard_fields.DetectionResultFields.detection_masks:
            np.pad(np.ones([1, 25, 25], dtype=np.uint8),
                   ((0, 0), (10, 10), (10, 10)), mode='constant')
        })
    metrics = coco_evaluator.evaluate()
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    coco_evaluator.clear()
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._detection_masks_list)


class CocoMaskEvaluationPyFuncTest(tf.test.TestCase):

  def testAddEvalDict(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }
    update_op = coco_evaluator.add_eval_dict(eval_dict)
    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.], [50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1, 2]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant')
                  ]),
              detection_scores:
                  np.array([.9, .8]),
              detection_classes:
                  np.array([2, 1]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant'),
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                  ])
          })
      self.assertLen(coco_evaluator._groundtruth_list, 2)
      self.assertLen(coco_evaluator._detection_masks_list, 2)

  def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    image_id = tf.placeholder(tf.string, shape=())
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(None))
    groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(None))
    detection_classes = tf.placeholder(tf.float32, shape=(None))
    detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id:
                  'image1',
              groundtruth_boxes:
                  np.array([[100., 100., 200., 200.], [50., 50., 100., 100.]]),
              groundtruth_classes:
                  np.array([1, 2]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant')
                  ]),
              detection_scores:
                  np.array([.9, .8]),
              detection_classes:
                  np.array([2, 1]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)),
                          mode='constant'),
                      np.pad(
                          np.ones([100, 100], dtype=np.uint8), ((10, 10),
                                                                (10, 10)),
                          mode='constant'),
                  ])
          })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image2',
                   groundtruth_boxes: np.array([[50., 50., 100., 100.]]),
                   groundtruth_classes: np.array([1]),
                   groundtruth_masks: np.pad(np.ones([1, 50, 50],
                                                     dtype=np.uint8),
                                             ((0, 0), (10, 10), (10, 10)),
                                             mode='constant'),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1]),
                   detection_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8),
                                           ((0, 0), (10, 10), (10, 10)),
                                           mode='constant')
               })
      sess.run(update_op,
               feed_dict={
                   image_id: 'image3',
                   groundtruth_boxes: np.array([[25., 25., 50., 50.]]),
                   groundtruth_classes: np.array([1]),
                   groundtruth_masks: np.pad(np.ones([1, 25, 25],
                                                     dtype=np.uint8),
                                             ((0, 0), (10, 10), (10, 10)),
                                             mode='constant'),
                   detection_scores: np.array([.8]),
                   detection_classes: np.array([1]),
                   detection_masks: np.pad(np.ones([1, 25, 25],
                                                   dtype=np.uint8),
                                           ((0, 0), (10, 10), (10, 10)),
                                           mode='constant')
               })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._detection_masks_list)

  def testGetOneMAPWithMatchingGroundtruthAndDetectionsBatched(self):
    coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
    batch_size = 3
    image_id = tf.placeholder(tf.string, shape=(batch_size))
    groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    groundtruth_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))
    detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
    detection_masks = tf.placeholder(
        tf.uint8, shape=(batch_size, None, None, None))

    input_data_fields = standard_fields.InputDataFields
    detection_fields = standard_fields.DetectionResultFields
    eval_dict = {
        input_data_fields.key: image_id,
        input_data_fields.groundtruth_boxes: groundtruth_boxes,
        input_data_fields.groundtruth_classes: groundtruth_classes,
        input_data_fields.groundtruth_instance_masks: groundtruth_masks,
        detection_fields.detection_scores: detection_scores,
        detection_fields.detection_classes: detection_classes,
        detection_fields.detection_masks: detection_masks,
    }

    eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)

    _, update_op = eval_metric_ops['DetectionMasks_Precision/mAP']

    with self.test_session() as sess:
      sess.run(
          update_op,
          feed_dict={
              image_id: ['image1', 'image2', 'image3'],
              groundtruth_boxes:
                  np.array([[[100., 100., 200., 200.]],
                            [[50., 50., 100., 100.]],
                            [[25., 25., 50., 50.]]]),
              groundtruth_classes:
                  np.array([[1], [1], [1]]),
              groundtruth_masks:
                  np.stack([
                      np.pad(
                          np.ones([1, 100, 100], dtype=np.uint8),
                          ((0, 0), (0, 0), (0, 0)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 50, 50], dtype=np.uint8),
                          ((0, 0), (25, 25), (25, 25)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 25, 25], dtype=np.uint8),
                          ((0, 0), (37, 38), (37, 38)),
                          mode='constant')
                  ],
                           axis=0),
              detection_scores:
                  np.array([[.8], [.8], [.8]]),
              detection_classes:
                  np.array([[1], [1], [1]]),
              detection_masks:
                  np.stack([
                      np.pad(
                          np.ones([1, 100, 100], dtype=np.uint8),
                          ((0, 0), (0, 0), (0, 0)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 50, 50], dtype=np.uint8),
                          ((0, 0), (25, 25), (25, 25)),
                          mode='constant'),
                      np.pad(
                          np.ones([1, 25, 25], dtype=np.uint8),
                          ((0, 0), (37, 38), (37, 38)),
                          mode='constant')
                  ],
                           axis=0)
          })
    metrics = {}
    for key, (value_op, _) in eval_metric_ops.items():
      metrics[key] = value_op
    metrics = sess.run(metrics)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'],
                           1.0)
    self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
    self.assertFalse(coco_evaluator._groundtruth_list)
    self.assertFalse(coco_evaluator._image_ids_with_detections)
    self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
    self.assertFalse(coco_evaluator._detection_masks_list)


if __name__ == '__main__':
  tf.test.main()
