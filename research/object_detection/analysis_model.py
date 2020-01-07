import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D as a3d

def Read_npy(path):
    return np.load(path)

def scores_filter(boxes, scores, threshold=0.5):
    """
    boxes: [num_boxes, 4]
    scores: [num_scores, 1]
    Note: num_boxes should be equal to num_scores
    """
    num_boxes = boxes.shape[0]
    num_scores = scores.shape[0]
    if num_boxes != num_scores:
        raise ValueError("The number of boxes is not equal to scores'")
    
    new_boxes = []
    new_scores = []
    for i in range(num_scores):
        if scores[i] < threshold:
            continue
        new_boxes.append(boxes[i, :])
        new_scores.append(scores[i])
    
    new_boxes = np.array(new_boxes)
    new_scores = np.array(new_scores)

    return new_boxes, new_scores

def Calculate_IOU(gt_boxes, predict_boxes, origin_width=1, origin_height=1):
    """ 
    Calculate the IOU between gt_boxes and predict_boxes
    gt_boxes: [xmin, ymin, xmax, ymax]
    predict_boxes: [ymin, xmin, ymax, xmax]
    """
    num_gt_boxes = gt_boxes.shape[0]
    num_predict_boxes = predict_boxes.shape[0]
    
    if num_gt_boxes == 0 or num_predict_boxes == 0:
        return np.array([0])
    
    IOUs = []
    for i in range(num_gt_boxes):
        if np.array_equal(gt_boxes[i,:], np.zeros([4])):
            continue
        
        # GT coordinates, Shape: (num_predict_boxes,)
        Xmin_gt = np.repeat(gt_boxes[i, 0], num_predict_boxes)
        Xmax_gt = np.repeat(gt_boxes[i, 2], num_predict_boxes)
        Ymin_gt = np.repeat(gt_boxes[i, 1], num_predict_boxes)
        Ymax_gt = np.repeat(gt_boxes[i, 3], num_predict_boxes)

        # Intersection coordinates, Shape: (num_predict_boxes,)
        Inter_Xmin = np.maximum(Xmin_gt, predict_boxes[:, 1] * origin_width)
        Inter_Xmax = np.minimum(Xmax_gt, predict_boxes[:, 3] * origin_width)
        Inter_Ymin = np.maximum(Ymin_gt, predict_boxes[:, 0] * origin_height)
        Inter_Ymax = np.minimum(Ymax_gt, predict_boxes[:, 2] * origin_height)

        # Squares of areas, Shape: (num_predict_boxes,)
        Inter_area = np.maximum((Inter_Ymax - Inter_Ymin), np.zeros(Inter_Ymax.shape)) * np.maximum((Inter_Xmax - Inter_Xmin), np.zeros(Inter_Xmax.shape))
        GT_area = (Ymax_gt - Ymin_gt) * (Xmax_gt - Xmin_gt)
        PD_area = (predict_boxes[:, 2] - predict_boxes[:, 0]) * (predict_boxes[:, 3] - predict_boxes[:, 1]) * origin_width * origin_height

        # IOU,  Shape: (num_predict_boxes,)
        IOU = Inter_area / (GT_area + PD_area - Inter_area)
        
        IOUs.append(IOU)

    IOUs = np.array(IOUs)

    return IOUs  # Shpae: (num_gt_boxes, num_predict_boxes), IOUs is a sparse matrix

def Calculate_GIOU(gt_boxes, predict_boxes, origin_width=1, origin_height=1):
     """ 
    Calculate the GIoU between gt_boxes and predict_boxes
    gt_boxes: [xmin, ymin, xmax, ymax]
    predict_boxes: [ymin, xmin, ymax, xmax]
    """
    num_gt_boxes = gt_boxes.shape[0]
    num_predict_boxes = predict_boxes.shape[0]
    
    if num_gt_boxes == 0 or num_predict_boxes == 0:
        return np.array([0])
    
    GIOUs = []
    IOUs = []
    for i in range(num_gt_boxes):
        if np.array_equal(gt_boxes[i,:], np.zeros([4])):
            continue
        
        # GT coordinates, Shape: (num_predict_boxes,)
        Xmin_gt = np.repeat(gt_boxes[i, 0], num_predict_boxes)
        Xmax_gt = np.repeat(gt_boxes[i, 2], num_predict_boxes)
        Ymin_gt = np.repeat(gt_boxes[i, 1], num_predict_boxes)
        Ymax_gt = np.repeat(gt_boxes[i, 3], num_predict_boxes)

        # Intersection coordinates, Shape: (num_predict_boxes,)
        Inter_Xmin = np.maximum(Xmin_gt, predict_boxes[:, 1] * origin_width)
        Inter_Xmax = np.minimum(Xmax_gt, predict_boxes[:, 3] * origin_width)
        Inter_Ymin = np.maximum(Ymin_gt, predict_boxes[:, 0] * origin_height)
        Inter_Ymax = np.minimum(Ymax_gt, predict_boxes[:, 2] * origin_height)

        # The smallest enclosing box C, Shape: (num_predict_boxes,)
        Enc_Xmin = np.minimum(Xmin_gt, predict_boxes[:, 1] * origin_width)
        Enc_Xmax = np.maximum(Xmax_gt, predict_boxes[:, 3] * origin_width)
        Enc_Ymin = np.minimum(Ymin_gt, predict_boxes[:, 0] * origin_height)
        Enc_Ymax = np.maximum(Ymax_gt, predict_boxes[:, 2] * origin_height)

        # Squares of areas, Shape: (num_predict_boxes,)
        Inter_area = np.maximum((Inter_Ymax - Inter_Ymin), np.zeros(Inter_Ymax.shape)) * np.maximum((Inter_Xmax - Inter_Xmin), np.zeros(Inter_Xmax.shape))
        Enc_area = np.maximum((Enc_Ymax - Enc_Ymin), np.zeros(Enc_Ymax.shape)) * np.maximum((Enc_Xmax - Enc_Xmin), np.zeros(Enc_Xmax.shape))
        GT_area = (Ymax_gt - Ymin_gt) * (Xmax_gt - Xmin_gt)
        PD_area = (predict_boxes[:, 2] - predict_boxes[:, 0]) * (predict_boxes[:, 3] - predict_boxes[:, 1]) * origin_width * origin_height

        # IOU,  Shape: (num_predict_boxes,)
        IOU = Inter_area / (GT_area + PD_area - Inter_area)
        
        # GIOU, shape: (num_predict_boxes,)
        GIOU = IOU - (Enc_area - (GT_area + PD_area - Inter_area)) / Enc_area
        
        GIOUs.append(GIOU)
        IOUs.append(IOU)

    GIOUs = np.array(GIOUs)
    IOUs = np.array(IOUs)

    return GIOUs, IOUs  # Shpae: (num_gt_boxes, num_predict_boxes), GIOUs and IOUs are a sparse matrix


if __name__ == "__main__":
    model_name = "320_240_batch_32_scale_0.05_0.4_warmup_cosine_data_aug"
    gt_boxes_dir = "E:\\realsense\\object_detection\\data2\\record_files\\coord_np_format\\test\\"
    pd_boxes_dir = "E:\\realsense\\object_detection\\data2\\record_files\\output\\test\\{}\\boxes\\".format(model_name)
    pd_scores_dir = "E:\\realsense\\object_detection\\data2\\record_files\\output\\test\\{}\\scores\\".format(model_name)
    img_dir = "E:\\realsense\\object_detection\\data2\\record_files\\img_np_format\\test\\"

    gt_boxes_list = os.listdir(gt_boxes_dir)
    pd_boxes_list = os.listdir(pd_boxes_dir)
    pd_scores_list = os.listdir(pd_scores_dir)
    img_list = os.listdir(img_dir)
    
    if len(gt_boxes_list) != len(pd_boxes_list) or len(pd_scores_list) != len(pd_boxes_list):
        raise RuntimeError("数据数量不匹配")

    # 目前正常来说，npy文件名都是以图像的文件名起的，而且顺序都一样，
    # 如果后面文件名不匹配或者顺序不一样，应该加上文件匹配的功能才行
    try:
        IOUs_Scores_list = []
        GIOUs_Scores_list = []
        IOUs_list = []
        GIOUs_list = []
        Scores_list = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(gt_boxes_list)):
            gt_boxes_path = os.path.join(gt_boxes_dir, gt_boxes_list[i])
            pd_boxes_path = os.path.join(pd_boxes_dir, pd_boxes_list[i])
            pd_scores_path = os.path.join(pd_scores_dir, pd_scores_list[i])
            img_path = os.path.join(img_dir, img_list[i])

            gt_boxes_np = Read_npy(gt_boxes_path)
            pd_boxes_np = Read_npy(pd_boxes_path)
            pd_scores_np = Read_npy(pd_scores_path)
            img_np = Read_npy(img_path)

            if gt_boxes_np.shape[0] == 1 and np.array_equal(gt_boxes_np[0],np.zeros(4)):
                continue 

            # for i in range(gt_boxes_np.shape[0]):
            #     xl = gt_boxes_np[i,0]
            #     yl = gt_boxes_np[i,1]
            #     xr = gt_boxes_np[i,2]
            #     yr = gt_boxes_np[i,3]
            #     cv2.rectangle(img_np, (int(xl), int(yl)), (int(xr), int(yr)), (0,255,0), 2)

            pd_boxes_np, pd_scores_np = scores_filter(pd_boxes_np, pd_scores_np, 0.1)
            
            if pd_boxes_np.shape[0] == 0:
                continue

            # for i in range(pd_boxes_np.shape[0]):
            #     xl = pd_boxes_np[i,1] * 320
            #     yl = pd_boxes_np[i,0] * 240
            #     xr = pd_boxes_np[i,3] * 320
            #     yr = pd_boxes_np[i,2] * 240
            #     cv2.rectangle(img_np, (int(xl), int(yl)), (int(xr), int(yr)), (0,0,255), 2)
            #     cv2.putText(img_np, str(pd_scores_np[i]), (int(xl), int(yl)), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
            GIOUs, IOUs = Calculate_GIOU(gt_boxes_np, pd_boxes_np, origin_width=320, origin_height=240)
            
            max_iou_vec = np.max(IOUs, axis=0)    # shape: (num_pd_boxes,)
            max_giou_vec = np.max(GIOUs, axis=0)  # shaep: (num_pd_boxes,)

            if isinstance(max_iou_vec, np.int32) and max_iou_vec == 0:
                continue

            if isinstance(max_giou_vec, np.int32) and max_giou_vec == 0:
                continue
            
            iou_scores = np.dstack((max_iou_vec, pd_scores_np))
            giou_scores = np.dstack((max_giou_vec, pd_scores_np))

            IOUs_Scores_list.append(iou_scores)
            GIOUs_Scores_list.append(giou_scores)
            IOUs_list.append(max_iou_vec)
            GIOUs_list.append(max_giou_vec)
            Scores_list.append(pd_scores_np)
            #plt.scatter(max_iou_vec, pd_scores_np, c='green', alpha=0.5)
            # cv2.imshow("img", img_np)
            # press_key = cv2.waitKey(0)
            # if press_key == 27:
            #     break
            print(i)
            # if i == 100:
            #     break
        GIOUs_np = np.hstack(GIOUs_list)
        IOUs_np = np.hstack(IOUs_list)
        Scores_np = np.hstack(Scores_list)

        # 3D histogram
        # hist, xedges, yedges = np.histogram2d(IOUs_np, Scores_np, bins=10, range=[[0, 1], [0, 1]])
        hist, xedges, yedges = np.histogram2d(GIOUs_np, Scores_np, bins=10, range=[[0, 1], [0, 1]])
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros(xpos.shape)

        dx = dy = 0.1 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        plt.xlabel("GIOU")
        plt.ylabel("Cls scores")
        # ax.plot_surface(xpos, ypos, zpos, rstride=1, cstride=1, cmap='rainbow')
        # plt.hist(IOUs_np, bins=10, range=[0,1], facecolor="blue", edgecolor="black", alpha=0.7, align='left')
        plt.show()
        
    except ValueError as e:
        print(gt_boxes_np.shape[0])
        print(e)