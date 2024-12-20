#!/usr/bin/env python3

import sys
sys.path.append('/home/a/tensor_ws/src/YOLOv8_Tracking/src/yolov8_deepsort/ultralytics/yolo/v8/detect/deep_sort_pytorch')

import rospy
import torch
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.checks import check_imgsz
from utils.parser import get_config
from deep_sort import DeepSort
from collections import deque
import numpy as np
from numpy import random
from pathlib import Path
import hydra
from ultralytics.utils.plotting import Annotator


# Global variables
bridge = CvBridge()
deepsort = None
data_deque = {}

# Initialize DeepSORT
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("/home/a/tensor_ws/src/YOLOv8_Tracking/src/yolov8_deepsort/ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

# Helper function to convert bounding box
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate the center of the bounding box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # 바운딩 박스의 중앙 계산
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = (0, 255, 0)  # Fixed green color for simplicity, customize as needed
        obj_name = names[object_id[i]]
        label = f'{id}:{obj_name}'
        data_deque[id].appendleft(center)

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw trail
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

        # Draw the tracker point at the center of the bounding box
        cv2.circle(img, center, 5, (0, 0, 255), -1)  # 원으로 중심점 표시

    return img



# DetectionPredictor ROS Wrapper
class DetectionPredictorROS:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        # YOLO 모델 로드
        self.model = YOLO(model_path)  # 모델 객체 생성
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.init_sub_pub()

    def init_sub_pub(self):
        self.image_sub = rospy.Subscriber("stab_image", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("tracking_image", Image, queue_size=1)

    def image_callback(self, msg):
        try:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        processed_image = self.process_frame(cv_image)

        try:
            tracking_msg = bridge.cv2_to_imgmsg(processed_image, encoding="bgr8")
            self.image_pub.publish(tracking_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # Normalize to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def process_frame(self, frame):
        orig_img = frame.copy()
        results = self.model(frame)  # Run model inference

        # Check if results contain detections
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        # If there are no detections, return the original image
        if result.boxes is None or len(result.boxes) == 0:
            rospy.loginfo("No detections found.")
            return orig_img

        # Get the bounding boxes, confidence scores, and class labels
        bbox_xyxy = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs (object classes)

        # Prepare bounding boxes in xywh format for DeepSORT
        xywh_bboxs = []
        confs = []
        oids = []

        for i, bbox in enumerate(bbox_xyxy):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*bbox)
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([confidences[i]])
            oids.append(int(class_ids[i]))

        # Convert to tensors
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # Update DeepSORT with the detections and the original image
        outputs = deepsort.update(xywhs, confss, oids, orig_img)

        # If DeepSORT returns tracking results, draw the boxes with IDs
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]  # Tracking IDs
            object_id = outputs[:, -1]  # Object class IDs
            draw_boxes(orig_img, bbox_xyxy, self.model.names, object_id, identities)

        # Display the image in a window
        cv2.imshow("Tracking Result", orig_img)  # 새로운 창에 트래킹된 이미지 표시
        cv2.waitKey(1)  # 1ms 대기, 계속해서 업데이트

        return orig_img



    def write_results(self, idx, preds, batch):
        det = preds[idx]
        if len(det) == 0:
            return

        xywh_bboxs, confs, oids = [], [], []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
            confs.append([conf.item()])
            oids.append(int(cls))

        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, batch)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(batch, bbox_xyxy, self.model.names, object_id, identities)


# Main Function
if __name__ == "__main__":
    rospy.init_node("yolo_deepsort_tracker", anonymous=True)
    init_tracker()

    cfg = {
    "model": "/home/a/tensor_ws/src/YOLOv8_Tracking/src/best.pt",
    "conf": 0.25,
    "iou": 0.45
    }

    predictor = DetectionPredictorROS(model_path=cfg["model"], conf_thresh=cfg["conf"], iou_thresh=cfg["iou"])
    rospy.spin()
