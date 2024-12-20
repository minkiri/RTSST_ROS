#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import rospkg
import threading
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from src.superpoint import SuperPointFrontend, PointTracker

class RTStitchNode:
    def __init__(self):
        rospy.init_node('rtstitch_node', anonymous=True)

        # ROS 이미지 토픽 구독
        self.image_sub1 = rospy.Subscriber('/usb_cam2/image_raw/compressed', CompressedImage, self.callback_camera1)
        self.image_sub2 = rospy.Subscriber('/usb_cam1/image_raw/compressed', CompressedImage, self.callback_camera2)

        # ROS 이미지 퍼블리셔
        self.image_pub = rospy.Publisher('/stitched_image/compressed', CompressedImage, queue_size=1)

        self.bridge = CvBridge()
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('RTStitch')

        model_path = f"{package_path}/resources/superpoint_v1.pth"
        weights_path = rospy.get_param('~weights_path', model_path)
        self.superpoint = SuperPointFrontend(
            weights_path=weights_path,
            nms_dist=4,
            conf_thresh=0.015,
            nn_thresh=0.7,
            cuda=torch.cuda.is_available()
        )

        self.frame1 = None
        self.frame2 = None

        self.homographies = []  # 호모그래피 저장 리스트
        self.avg_H = None  # 평균 호모그래피
        self.frame_count = 0
        self.initial_frames = 10  # 초기 평균 계산할 프레임 수
        self.stabilized_H = None  # 고정 호모그래피
        self.use_fixed_H = False  # 고정 호모그래피 사용 플래그

    def resize_frames(self, frame, scale=0.7):
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    def callback_camera1(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.frame1 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error processing frame from camera 1: {e}")

    def callback_camera2(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.frame2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            rospy.logerr(f"Error processing frame from camera 2: {e}")

    def process_frames(self):
        if self.frame1 is None or self.frame2 is None:
            rospy.loginfo("Waiting for frames from both cameras...")
            return

        self.frame_count += 1

        if not self.use_fixed_H:
            # 초기화 단계에서 고정 호모그래피를 계산
            gray1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gray2 = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            pts1, desc1, _ = self.superpoint.run(gray1)
            pts2, desc2, _ = self.superpoint.run(gray2)

            tracker = PointTracker(max_length=5, nn_thresh=self.superpoint.nn_thresh)
            matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.superpoint.nn_thresh)

            if matches.shape[1] >= 4:  # 최소 4개의 대응점 필요
                pts1_matched = pts1[:2, matches[0, :].astype(int)].T
                pts2_matched = pts2[:2, matches[1, :].astype(int)].T
                H, _ = cv2.findHomography(pts2_matched, pts1_matched, cv2.RANSAC, ransacReprojThreshold=2)

                if H is not None:
                    self.homographies.append(H)

                    if self.frame_count == self.initial_frames:
                        # 고정된 호모그래피 계산
                        self.stabilized_H = np.mean(self.homographies, axis=0)
                        self.stabilized_H /= self.stabilized_H[2, 2]  # 정규화
                        self.use_fixed_H = True
                        rospy.loginfo("Stabilized homography calculated and fixed.")
                else:
                    rospy.logwarn("Homography calculation failed. Skipping frame.")
                    return
            else:
                rospy.logwarn("Not enough points for homography. Skipping frame.")
                return
        else:
            # 고정된 호모그래피를 사용해 스티칭
            if self.stabilized_H is not None:
                self.stitch_frames(self.stabilized_H)
            else:
                rospy.logwarn("Stabilized homography is not available.")

    def blend_linear(self, warp_img1, warp_img2):
        """
        두 이미지를 선형적으로 블렌딩
        """
        img1 = warp_img1
        img2 = warp_img2

        # 이미지 마스크 생성
        img1mask = ((img1[:, :, 0] | img1[:, :, 1] | img1[:, :, 2]) > 0)
        img2mask = ((img2[:, :, 0] | img2[:, :, 1] | img2[:, :, 2]) > 0)

        # 각 이미지의 중심 좌표 계산
        r, c = np.nonzero(img1mask)
        out_1_center = [np.mean(r), np.mean(c)]
        r, c = np.nonzero(img2mask)
        out_2_center = [np.mean(r), np.mean(c)]

        # 중심 벡터 계산
        vec = np.array(out_2_center) - np.array(out_1_center)
        intsct_mask = img1mask & img2mask

        r, c = np.nonzero(intsct_mask)
        out_wmask = np.zeros(img2mask.shape[:2])

        # 겹치는 부분의 선형 블렌딩 값 계산
        proj_val = (r - out_1_center[0]) * vec[0] + (c - out_1_center[1]) * vec[1]
        out_wmask[r, c] = (proj_val - (min(proj_val) + 1e-3)) / \
                          ((max(proj_val) - 1e-3) - (min(proj_val) + 1e-3))

        # 마스크 생성
        mask1 = img1mask & (out_wmask == 0)
        mask2 = out_wmask
        mask3 = img2mask & (out_wmask == 0)

        # 블렌딩 수행
        out = np.zeros(img1.shape)
        for c in range(3):
            out[:, :, c] = img1[:, :, c] * (mask1 + (1 - mask2) * (mask2 != 0)) + \
                           img2[:, :, c] * (mask2 + mask3)

        return np.uint8(out)

    def stitch_frames(self, H):
        """
        스티칭 수행 함수: 고정된 호모그래피 사용
        """
        height1, width1 = self.frame1.shape[:2]
        height2, width2 = self.frame2.shape[:2]

        max_width = width1 + width2
        max_height = max(height1, height2)

        # 캔버스 초기화
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        canvas[:height1, :width1] = self.frame1

        # 두 번째 이미지를 고정된 호모그래피로 변환
        warped_frame2 = cv2.warpPerspective(self.frame2, H, (max_width, max_height))

        # 블렌딩
        blended_canvas = self.blend_linear(canvas, warped_frame2)

        # 스티칭된 이미지 퍼블리시
        try:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', blended_canvas)[1]).tobytes()
            self.image_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"Error publishing stitched image: {e}")

    def process_thread(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            self.process_frames()
            rate.sleep()

    def run(self):
        thread = threading.Thread(target=self.process_thread)
        thread.start()
        rospy.spin()


if __name__ == "__main__":
    try:
        node = RTStitchNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
