import torch
import cv2
import numpy as np
from model.path_smooth_net import PathSmoothUNet
from collections import OrderedDict
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 사전 학습된 모델 파일 경로
model_path = '/home/a/tensor_ws/src/RTStab/src/NNDVS/pretrained/pretrained_model.pth.tar'

# 모델 로드 및 설정
net_radius = 15
model = PathSmoothUNet(in_chn=4 * net_radius)
model = model.cuda()

# 체크포인트 로드 및 'module.' 접두사 제거
checkpoint = torch.load(model_path, map_location='cuda')
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# 실시간 웹캠 입력 설정
video_capture = cv2.VideoCapture(0)
scale_factor = 4
bilinear_upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

# 프레임 스택 생성
frame_stack = []
transform = transforms.ToTensor()
stabilization_strength = 2.0  # 흔들림 보정 강도 조절

# 흐름 필드 필터링을 위한 Gaussian 블러 개선
def smooth_flow(flow, kernel_size=5, sigma=1.5):
    flow_smoothed = cv2.GaussianBlur(flow, (kernel_size, kernel_size), sigma)
    flow_smoothed[:, :, 0] = np.clip(flow_smoothed[:, :, 0], -5, 5)
    flow_smoothed[:, :, 1] = np.clip(flow_smoothed[:, :, 1], -5, 5)
    return flow_smoothed

# 흐름 필드의 이동 범위 제한
def clip_flow(flow, max_movement=10):
    flow[:, :, 0] = np.clip(flow[:, :, 0], -max_movement, max_movement)
    flow[:, :, 1] = np.clip(flow[:, :, 1], -max_movement, max_movement)
    return flow

# PSNR 계산 함수
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:  # 영상이 완전히 동일하면 PSNR은 무한대
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

# SSIM 계산 함수
def calculate_ssim(image1, image2):
    # 이미지를 그레이스케일로 변환
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # SSIM 계산 시 win_size 값을 3 또는 5로 설정
    return ssim(gray1, gray2, win_size=3, multichannel=False)

# matplotlib 그래프 초기화
fig, (ax_psnr, ax_ssim) = plt.subplots(2, 1, figsize=(10, 12))  # 두 개의 그래프 생성

# 지표 값들을 기록할 리스트
psnr_values = []
ssim_values = []

# 그래프 설정 (PSNR)
ax_psnr.set_xlim(0, 100)  # 최근 100프레임만 표시
ax_psnr.set_ylim(0, 50)  # PSNR 값 범위 (0~50)
ax_psnr.set_ylabel("PSNR Value")
ax_psnr.set_xlabel("Frame ID")

# PSNR 값에 해당하는 선을 추가
line_psnr, = ax_psnr.plot([], [], color='green', label="PSNR")
ax_psnr.legend()

# 그래프 설정 (SSIM)
ax_ssim.set_xlim(0, 100)  # 최근 100프레임만 표시
ax_ssim.set_ylim(0, 1)  # SSIM 값 범위 (0~1)
ax_ssim.set_ylabel("SSIM Value")
ax_ssim.set_xlabel("Frame ID")

# SSIM 값에 해당하는 선을 추가
line_ssim, = ax_ssim.plot([], [], color='red', label="SSIM")
ax_ssim.legend()

# 실시간 비디오 처리
frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    original_frame = frame.copy()
    
    # 모델 해상도에 맞춰 크기 조정
    frame_resized = cv2.resize(frame, (640, 480))
    frame_tensor = transform(frame_resized).cuda()

    # 프레임 스택에 추가
    for channel in frame_tensor:
        frame_stack.append(channel.unsqueeze(0))

    if len(frame_stack) > 4 * net_radius:
        frame_stack = frame_stack[-4 * net_radius:]

    if len(frame_stack) == 4 * net_radius:
        stacked_tensor = torch.cat(frame_stack, dim=0).unsqueeze(0)

        with torch.no_grad():
            stabilized_tensor = model(stacked_tensor)
            flow_field = bilinear_upsample(stabilized_tensor).squeeze().permute(1, 2, 0).cpu().numpy()

            # 흐름 필드를 원본 크기로 리샘플링
            flow_field_resized = cv2.resize(flow_field, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_CUBIC)

            # 흐름 필드를 부드럽게 처리
            flow_field_resized = smooth_flow(flow_field_resized)

            # 흐름 필드의 이동 범위를 제한
            flow_field_resized = clip_flow(flow_field_resized)

            # 흐름 필드에 강도 적용
            flow_field_resized *= stabilization_strength

            # 흐름 필드를 활용하여 원본 프레임 보정
            h, w = original_frame.shape[:2]
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = map_x.astype(np.float32) + flow_field_resized[:, :, 0]
            map_y = map_y.astype(np.float32) + flow_field_resized[:, :, 1]

            stabilized_frame = cv2.remap(original_frame, map_x, map_y, cv2.INTER_LINEAR)

            # PSNR 계산
            psnr_value = calculate_psnr(original_frame, stabilized_frame)

            # SSIM 계산
            ssim_value = calculate_ssim(original_frame, stabilized_frame)

            # 결과 저장
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            # 그래프 데이터 업데이트
            if len(psnr_values) > 100:
                psnr_values.pop(0)
                ssim_values.pop(0)

            # 선 데이터 설정 (PSNR)
            line_psnr.set_data(range(len(psnr_values)), psnr_values)
            # 선 데이터 설정 (SSIM)
            line_ssim.set_data(range(len(ssim_values)), ssim_values)

            ax_psnr.relim()
            ax_psnr.autoscale_view()

            ax_ssim.relim()
            ax_ssim.autoscale_view()

            plt.draw()
            plt.pause(0.01)

            # 원본과 보정된 프레임을 나란히 보여주기
            combined_frame = cv2.hconcat([original_frame, stabilized_frame])
            cv2.imshow("Combined Frame", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

