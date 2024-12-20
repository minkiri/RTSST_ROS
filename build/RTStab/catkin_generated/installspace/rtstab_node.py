import sys
sys.path.append('/home/a/tensor_ws/src/RTStab/src')

import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from torchvision import transforms
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
from model.path_smooth_net import PathSmoothUNet  # 모델 임포트

# 모델 설정
model_path = '/home/a/tensor_ws/src/RTStab/src/NNDVS/pretrained/pretrained_model.pth.tar'
net_radius = 15  # 최신 15프레임 스택
model = PathSmoothUNet(in_chn=3 * net_radius).cuda()

# 모델 가중치 로드
checkpoint = torch.load(model_path, map_location='cuda')
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model.eval()
model = model.half()  # 반정밀도로 설정

# ROS 설정
rospy.init_node('rtstab_node')
bridge = CvBridge()
image_pub = rospy.Publisher("/stab_image", Image, queue_size=10)
image_subscriber = rospy.Subscriber("/stitched_image/compressed", CompressedImage, lambda msg: process_image(msg))

# 입력 이미지 처리 설정
transform = transforms.ToTensor()
frame_stack = []

# 흐름 매개변수 최적화
def smooth_flow(flow, kernel_size=5, sigma=1.5):
    flow = cv2.GaussianBlur(flow.astype(np.float32), (kernel_size, kernel_size), sigma)
    flow[:, :, 0] = np.clip(flow[:, :, 0], -5, 5)
    flow[:, :, 1] = np.clip(flow[:, :, 1], -5, 5)
    return flow

# PSNR 계산
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

# SSIM 계산
def calculate_ssim(image1, image2):
    gray1, gray2 = map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (image1, image2))
    return ssim(gray1, gray2, win_size=3, multichannel=False)

# ROS 메시지 처리
def process_image(msg):
    global frame_stack

    # ROS 이미지 -> OpenCV 이미지 변환
    frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    frame_resized = cv2.resize(frame, (320, 240))
    frame_tensor = transform(frame_resized).unsqueeze(0).half().cuda()

    frame_stack.append(frame_tensor)
    if len(frame_stack) > net_radius:
        frame_stack = frame_stack[-net_radius:]

    if len(frame_stack) == net_radius:
        # 모델 입력 생성 (60채널)
        stacked_tensor = torch.cat(frame_stack, dim=1)

        try:
            with torch.no_grad():
                stabilized_tensor = model(stacked_tensor)

                for i in range(stabilized_tensor.size(0)):
                    flow_field = stabilized_tensor[i].permute(1, 2, 0).cpu().numpy()
                    flow_field = smooth_flow(flow_field)
                    h, w = frame_resized.shape[:2]
                    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x, map_y = map_x.astype(np.float32) + flow_field[:, :, 0], map_y.astype(np.float32) + flow_field[:, :, 1]

                    stabilized_frame = cv2.remap(frame_resized, map_x, map_y, cv2.INTER_LINEAR)

                    psnr_value = calculate_psnr(frame_resized, stabilized_frame)
                    ssim_value = calculate_ssim(frame_resized, stabilized_frame)

                    rospy.loginfo(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

                    # ROS 메시지 발행
                    stab_image_msg = bridge.cv2_to_imgmsg(stabilized_frame, encoding="bgr8")
                    image_pub.publish(stab_image_msg)

        except Exception as e:
            rospy.logerr(f"Error during stabilization: {e}")

        # GPU 메모리 관리
        torch.cuda.empty_cache()

rospy.spin()

