import sys
sys.path.append('/home/a/tensor_ws/src/RTStab/src')

import rospy
import torch
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from torchvision import transforms
from NNDVS.model.path_smooth_net import PathSmoothUNet
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim

# 모델 경로 및 설정
model_path = '/home/a/tensor_ws/src/RTStab/src/NNDVS/pretrained/pretrained_model.pth.tar'
net_radius = 15

# 모델 정의 (4 * net_radius 채널로 입력 크기 설정)
model = PathSmoothUNet(in_chn=4 * net_radius).cuda()

# 체크포인트 로드
checkpoint = torch.load(model_path, map_location='cuda')
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()

# state_dict에서 "module." 접두사 제거 후 새 dict로 저장
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v

# 모델 파라미터 로드
model.load_state_dict(new_state_dict)

# 모델을 평가 모드로 설정
model.eval()

# 반정밀도(half precision) 모델로 변환
model = model.half()

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True  # 성능 최적화
batch_size = 1  # 배치 크기 1로 설정

rospy.init_node('rtstab_node')

bridge = CvBridge()
transform = transforms.ToTensor()
stabilization_strength = 1.0

image_pub = rospy.Publisher("/stab_image", Image, queue_size=10)

frame_stack = []

# 프레임 스택 크기 제한
frame_stack_size = 5  # 프레임 스택 크기 줄이기

# 흐름 매개변수 최적화
def smooth_flow(flow, kernel_size=5, sigma=1.5):
    flow = flow.astype(np.float32)
    flow_smoothed = cv2.GaussianBlur(flow, (kernel_size, kernel_size), sigma)
    flow_smoothed[:, :, 0] = np.clip(flow_smoothed[:, :, 0], -5, 5)
    flow_smoothed[:, :, 1] = np.clip(flow_smoothed[:, :, 1], -5, 5)
    return flow_smoothed

def clip_flow(flow, max_movement=10):
    flow[:, :, 0] = np.clip(flow[:, :, 0], -max_movement, max_movement)
    flow[:, :, 1] = np.clip(flow[:, :, 1], -max_movement, max_movement)
    return flow

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def calculate_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, win_size=3, multichannel=False)

def image_callback(msg):
    global frame_stack

    # 압축된 이미지를 OpenCV 이미지로 변환
    frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

    # 원본 크기 유지
    frame_resized = frame

    # 이미지를 텐서로 변환하고 half precision 사용
    frame_tensor = transform(frame_resized).unsqueeze(0).half().cuda()  

    # 채널 수 맞추기 (60 채널로 확장)
    if frame_tensor.shape[1] != 60:
        repeat_count = 60 // frame_tensor.shape[1]
        frame_tensor = frame_tensor.repeat(1, repeat_count, 1, 1)  # 채널을 60으로 맞추기

    # 배치 크기 1로 설정
    frame_stack.append(frame_tensor)

    if len(frame_stack) > 1:  # 매 프레임마다 처리하도록 수정
        # 배치 크기 1로 처리
        frame_tensor = frame_stack[-1]  # 가장 최근 프레임만 사용
        frame_stack = []

        try:
            with torch.no_grad():
                # GPU에서 처리
                stabilized_tensor = model(frame_tensor)
                
                # 모델 출력 확인
                print(f"Model output shape: {stabilized_tensor.shape}")
                
                flow_field = stabilized_tensor[0].permute(1, 2, 0).cpu().numpy()

                # 흐름 필드 처리 및 이미지 리사이징
                flow_field_resized = smooth_flow(flow_field)
                flow_field_resized = clip_flow(flow_field_resized)
                flow_field_resized *= stabilization_strength

                h, w = frame_resized.shape[:2]
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = map_x.astype(np.float32) + flow_field_resized[:, :, 0]
                map_y = map_y.astype(np.float32) + flow_field_resized[:, :, 1]

                # CPU에서 remap 수행
                stabilized_frame = cv2.remap(frame_resized, map_x, map_y, interpolation=cv2.INTER_LINEAR)

                psnr_value = calculate_psnr(frame_resized, stabilized_frame)
                ssim_value = calculate_ssim(frame_resized, stabilized_frame)

                rospy.loginfo("PSNR: %.2f, SSIM: %.4f", psnr_value, ssim_value)

                # 이미지를 ROS 형식으로 변환하여 발행
                stab_image_msg = bridge.cv2_to_imgmsg(stabilized_frame, encoding="bgr8")
                image_pub.publish(stab_image_msg)

        except Exception as e:
            rospy.logerr(f"모델 처리 중 오류 발생: {e}")

        # GPU 메모리 최적화
        torch.cuda.empty_cache()

# ROS 이미지 구독자
image_subscriber = rospy.Subscriber("/stitched_image/compressed", CompressedImage, image_callback)

# ROS 노드 실행
rospy.spin()

