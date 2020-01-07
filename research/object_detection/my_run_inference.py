import numpy as np
import cv2
import os
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from my_ext_tfrecord import read_tfrecord, _decode_tfrecord_fn, _decode_tfrecord_only_img_fn


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

PATH_TO_LABELS = 'E:\\realsense\\object_detection\\annotation\\head_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_model_1_x(Frozen_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(Frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

if __name__ == "__main__":
    img_width = 320
    img_height = 240
    netInput_width = 300
    netInput_height = 300
    model_name = "320_240_batch32_scale_0.05_0.8_warmup_cosine_data_aug_2"
    FM_path = "E:\\realsense\\object_detection\\inference_graph\\{}\\frozen_inference_graph.pb".format(model_name)
    TF_path = "E:\\realsense\\object_detection\\data2\\record_files\\record\\test.record"
    save_data_path = "E:\\realsense\\object_detection\\data2\\record_files\\"

    output_key = ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']
    tensor_keys = []

    # Load model
    Inf_graph = load_model_1_x(FM_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False,
                               gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0))
    
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default() as tfrecord_graph:
        # Create Data pipeline
        se = read_tfrecord(TF_path)
        dataset = se.map(_decode_tfrecord_fn)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        next_element_img = tf.identity(tf.reshape(next_element[0], [netInput_height, netInput_width, 3]), name="next_element_img")
        next_element_filename = tf.identity(next_element[1], name="next_element_filename")
    
    tfrecord_graph_def = tfrecord_graph.as_graph_def()

    with Inf_graph.as_default() as inference_graph:
        # Create input tensor placehold and output dict tensor
        all_node_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in output_key:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
                tensor_keys.append(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, img_height, img_width)
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        
        # NOTE:
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # 如果直接用tfrecord_graph的next_element_img的节点feed给image_tensor，会出现以下错误：
        #       Invalid argument: input must be 4-dimensional[1,1,300,300,3]
        # 如果直接sess.run(next_element_img)的话，会返回一个shape为[1,h,w,c]的nparray，
        # 所以我认为是把next_element_img的tensor节点直接feed给image_tensor时候，会再增加一个维度，变成了[1,1,h,w,c]
        # input_tensor = tf.placeholder(dtype=tf.uint8,shape=[1, 1, netInput_height, netInput_width, 3], name="netInputTensor")
        # image_tensor = input_tensor[0]

    inference_graph_def = inference_graph.as_graph_def()

    with tf.Graph().as_default() as combine_graph:
        # import tfrecord extracting graph
        next_element_img_def = tf.import_graph_def(tfrecord_graph_def, return_elements=["next_element_img:0"])
        next_element_filename_def = tf.import_graph_def(tfrecord_graph_def, return_elements=["next_element_filename:0"])
        # import inference graph:
        # boxes_output = tf.import_graph_def(inference_graph_def, input_map={"image_tensor:0": next_element_img_def}, 
        #                                 return_elements=["detection_boxes:0"])
        # scores_output = tf.import_graph_def(inference_graph_def, input_map={"image_tensor:0": next_element_img_def}, 
        #                                 return_elements=["detection_scores:0"])

        inf_output = tf.import_graph_def(inference_graph_def, input_map={"image_tensor:0": next_element_img_def}, 
                                        return_elements=tensor_keys)
        with tf.Session() as sess:
            try:
                while True:
                    #out_imgs, out_filename, out_points = sess.run(next_element)
                    [output_dict, filename, img] = sess.run([inf_output, next_element_filename_def, next_element_img_def])
                    str_filename = filename[0].decode("utf-8")
                    print(str_filename)
                    out_img = img[0]
                    boxes = output_dict[1][0]     # [100, 4] 稀疏矩阵，
                    scores = output_dict[2][0]       # [100, ] 稀疏矩阵,从大到小排序
                    for i in range(boxes.shape[0]):
                        if(np.array_equal(boxes[i], np.zeros([4]))):
                            continue
                        if(scores[i]<0.4):
                            continue
                        # [ymin, xmin, ymax, xmax]
                        yl = boxes[i][0] * netInput_width
                        xl = boxes[i][1] * netInput_height
                        yr = boxes[i][2] * netInput_width
                        xr = boxes[i][3] * netInput_height
                        cv2.rectangle(out_img, (int(xl), int(yl)), (int(xr), int(yr)), (0,255,0), 2)
                        cv2.putText(out_img, str(scores[i]), (int(xl), int(yl)), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                    # cv2.imshow("img", out_img)
                    # press_key = cv2.waitKey(0)
                    # if press_key == 27:
                    #     break
                    np.save(os.path.join(os.path.join(save_data_path, "output\\test\\{}\\boxes\\".format(model_name)), 
                                            "{}.npy".format(str_filename.split(".")[0])), boxes)
                    np.save(os.path.join(os.path.join(save_data_path, "output\\test\\{}\\scores\\".format(model_name)), 
                                            "{}.npy".format(str_filename.split(".")[0])), scores)
            except tf.errors.OutOfRangeError:
                print("Finished")