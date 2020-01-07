# 检验tfrecord是否正确
import cv2
import os
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt

# Feature List
features_list = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
}

def read_tfrecord(tfr_path):
    return tf.data.TFRecordDataset(tfr_path)

def _img_decode2cv_py_fn(img_byte_list): 
    nparr = np.fromstring(img_byte_list, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv_img

def _img_decode2cv_withResize_py_fn(img_byte_list, target_width, target_height):
    nparr = np.fromstring(img_byte_list, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv_img = cv2.resize(cv_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return cv_img

def _zip_as_points_py_fn(xmins, xmaxs, ymins, ymaxs):
    return np.dstack((xmins, ymins, xmaxs, ymaxs))

def _resize_img_map_fn(args):
    # called by tf.map_fn
    img = args[0]
    target_size = args[1]
    return tf.image.resize_bilinear(img, target_size)

def _decode_tfrecord_fn(serialized_example, do_resize=True, target_width=300, target_height=300):
    data = tf.io.parse_single_example(serialized_example, features_list)
    filename_list = data['image/filename']
    img_byte_list = data['image/encoded']
    xmin_list = data['image/object/bbox/xmin']
    xmax_list = data['image/object/bbox/xmax']
    ymin_list = data['image/object/bbox/ymin']
    ymax_list = data['image/object/bbox/ymax']

    xmin_list = tf.sparse_tensor_to_dense(xmin_list)
    xmax_list = tf.sparse_tensor_to_dense(xmax_list)
    ymin_list = tf.sparse_tensor_to_dense(ymin_list)
    ymax_list = tf.sparse_tensor_to_dense(ymax_list)

    if do_resize:
        img = tf.py_func(_img_decode2cv_withResize_py_fn, [img_byte_list, target_width, target_height], [tf.uint8])
    else:
        img = tf.py_func(_img_decode2cv_py_fn, [img_byte_list], [tf.uint8])
    points = tf.py_func(_zip_as_points_py_fn, [xmin_list, xmax_list, ymin_list, ymax_list], [tf.float32])

    return img, filename_list, points

def _decode_tfrecord_only_img_fn(serialized_example, do_resize=True, target_width=300, target_height=300):
    data = tf.io.parse_single_example(serialized_example, features_list)
    img_byte_list = data['image/encoded']
    img = tf.py_func(_img_decode2cv_py_fn, [img_byte_list], [tf.uint8])
    # if do_resize:
    #     img = tf.py_func(_resize_img, [img, target_height, target_width], [tf.uint8])
    return img
    
if __name__ == '__main__':
    tf_path = "E:\\realsense\\object_detection\\data2\\record_files\\record\\test.record"
    save_data_path = "E:\\realsense\\object_detection\\data2\\record_files\\"
    print("Reading record...")
    se = read_tfrecord(tf_path)
    dataset = se.map(_decode_tfrecord_fn)
    iterator = dataset.make_one_shot_iterator()
    it = iterator.get_next()
    print("Start to extract tfrecord...")
    with tf.Session() as sess:
        try:
            width_vector = []
            height_vector = []
            relative_width_vector = []
            relative_height_vector = []
            imgs_batch = []
            batch_size = 8
            batch_size_cnt = 0
            num_batch_cnt = 0
            img_cnt = 0
            while True:
                img_cnt += 1
                out_imgs, out_filename, out_points = sess.run(it)
                pts = out_points[0][0]
                out_img = out_imgs[0]
                str_filename = out_filename.decode("utf-8") 
                coord_vector = []
                print("File name: {}".format(str_filename))
                #print("Point: {};".format(pts))
                # if str_filename.find('box') == -1:
                #     continue
                # for i in range(pts.shape[0]):
                #     xl = int(pts[i, 0] * 320)
                #     xr = int(pts[i, 2] * 320)
                #     yl = int(pts[i, 1] * 240)
                #     yr = int(pts[i, 3] * 240)
                #     relative_width_vector.append(pts[i, 2] - pts[i, 0])
                #     relative_height_vector.append(pts[i, 3] - pts[i, 1])
                #     width_vector.append(xr - xl)
                #     height_vector.append(yr - yl)
                #     coord_vector.append([xl, yl, xr, yr])
                    #cv2.rectangle(out_img, (xl, yl), (xr, yr), (0,0,255), 2)
                # np_coord_vector = np.array(coord_vector)
                # np_img_mat = np.asarray(out_img[:,:])
                # np.save(os.path.join(os.path.join(save_data_path, "img_np_format\\test"), 
                #                         "{}.npy".format(str_filename.split(".")[0])), np_img_mat)
                # np.save(os.path.join(os.path.join(save_data_path, "coord_np_format\\test"), 
                #                         "{}.npy".format(str_filename.split(".")[0])), np_coord_vector)
                # if batch_size_cnt < batch_size:
                #     imgs_batch.append(np_img_mat)
                #     batch_size_cnt += 1
                # else:
                #     np_imgs_batch = np.array(imgs_batch)
                #     np.save(os.path.join(os.path.join(save_data_path, "img_np_format\\test"), 
                #                         "img_mat_{}.npy".format(num_batch_cnt)), np_imgs_batch)
                #     print("Saved batch {}...".format(num_batch_cnt))
                #     num_batch_cnt += 1            
                #     imgs_batch.clear()
                #     imgs_batch.append(np.asarray(out_img[:,:]))
                #     batch_size_cnt = 1

                #cv2.imshow('img', out_img)
                #press_key = cv2.waitKey(0)
                #if press_key == 27:
                    #break
                # TODO: 得到filename后对比回标注文件，看看bbox是否标注正确
        except tf.errors.OutOfRangeError:
            # np_width_vector = np.array(width_vector)
            # np_height_vector = np.array(height_vector)
            # np_relative_width_vector = np.array(relative_width_vector)
            # np_relative_height_vector = np.array(relative_height_vector)
            # np_wh_vector = np.dstack((np_width_vector, np_height_vector))
            # np_relative_wh_vector = np.dstack((np_relative_width_vector, np_relative_height_vector))
            # np.save(os.path.join(os.path.join(save_data_path, "data_anaylsis\\test"), "wh_data.npy"), np_wh_vector)
            # np.save(os.path.join(os.path.join(save_data_path, "data_anaylsis\\test"), "re_wh_data.npy"), np_relative_wh_vector)
            #ratio_vector = np_width_vector / np_height_vector

            print("Total {} images".format(img_cnt))

            # plt.figure(1)
            # plt.hist(ratio_vector, bins=60)
            # plt.xlim(0,6)
            # plt.xlabel("W / H Ratio")
            # plt.ylabel("Freq")
            # plt.savefig(save_data_path + "ratios.png")

            # plt.figure(2)
            # plt.hist(relative_width_vector, bins=100)
            # plt.xlim(0,1)
            # plt.xlabel("Relative Scale")
            # plt.ylabel("Freq")
            # plt.savefig(save_data_path + "relative_scale.png")

