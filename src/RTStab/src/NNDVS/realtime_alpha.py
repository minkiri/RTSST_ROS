import torch
import cv2
import numpy as np
from model.path_smooth_net import PathSmoothUNet
from collections import OrderedDict
from torchvision import transforms
from PIL import Image

# 사전 학습된 모델 파일 경로
model_path = '/home/a/tensor_ws/src/RTStab/src/NNDVS/pretrained/pretrained_model.pth.tar'

# 모델 로드 및 설정
net_radius = 20  # net_radius = 20으로 설정하여 입력이 정확히 60채널이 되도록 맞춤
model = PathSmoothUNet(in_chn=3 * net_radius)  # 입력 채널 수를 60으로 설정
model = model.cuda()  # GPU 사용 설정

# 체크포인트 로드 및 'module.' 접두사 제거
checkpoint = torch.load(model_path, map_location='cuda', weights_only=True)
state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # 'module.' 제거
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()  # 평가 모드로 설정

# 실시간 웹캠 입력 설정
video_capture = cv2.VideoCapture(0)  # 기본 웹캠 사용
scale_factor = 2  # 업샘플링 비율 줄임
bilinear_upsample = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

# 프레임 스택 생성
frame_stack = []

# 직접 ToTensor 구현 (PIL -> Tensor 변환)
def to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB로 변환
    frame = np.transpose(frame, (2, 0, 1))  # [H, W, C] -> [C, H, W]로 변환
    frame_tensor = torch.from_numpy(frame).float() / 255.0  # [0, 1] 범위로 정규화 후 Tensor로 변환
    print("Tensor shape after conversion:", frame_tensor.shape)  # 디버깅 출력
    return frame_tensor

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # 해상도 줄이기
    frame = cv2.resize(frame, (320, 240))  # 해상도를 320x240으로 줄임
    original_frame = frame.copy()  # 원본 컬러 프레임을 저장

    # 직접 변환
    frame_tensor = to_tensor(frame).cuda()  # [3, H, W] 형태로 변환 후 GPU로 이동

    # 프레임 스택에 추가하고 60채널 유지
    frame_stack.append(frame_tensor)
    if len(frame_stack) > net_radius:
        frame_stack = frame_stack[-net_radius:]  # 최신 20개의 프레임만 유지

    # 스택의 각 프레임을 채널로 쌓아 60채널 입력 구성
    if len(frame_stack) == net_radius:
        # frame_stack의 각 프레임이 [3, H, W] 형태이므로 이를 60채널로 합침
        print("Stacked tensor shape before cat:", frame_stack[0].shape)  # 디버깅 출력
        
        # 각 프레임을 (1, 3, H, W)로 변환 후, 채널 차원(dim=1)으로 합침
        stacked_tensor = torch.cat(frame_stack, dim=0).unsqueeze(0)  # 배치 차원 추가
        print("Stacked tensor shape after cat:", stacked_tensor.shape)  # 디버깅 출력

        # 모델을 통한 흔들림 보정
        with torch.no_grad():
            stabilized_tensor = model(stacked_tensor)
            stabilized_tensor = bilinear_upsample(stabilized_tensor)

            # 2채널 결과를 3채널로 확장하여 컬러로 변환
            stabilized_tensor = stabilized_tensor.squeeze().cpu().numpy()
            if stabilized_tensor.shape[0] == 2:
                # 3채널로 확장
                stabilized_color = cv2.merge([stabilized_tensor[0], stabilized_tensor[1], stabilized_tensor[1]])
            else:
                stabilized_color = stabilized_tensor

            # stabilized_color의 크기가 원본 프레임과 다를 수 있으므로 원본 프레임 크기에 맞춰 리사이즈
            stabilized_color = cv2.resize(stabilized_color, (original_frame.shape[1], original_frame.shape[0]))

            # 값 범위를 0-255로 조정
            stabilized_color = np.clip(stabilized_color, 0, 255).astype('uint8')

            # 컬러 이미지를 원본과 결합하여 출력
            stabilized_frame = cv2.addWeighted(original_frame, 0.5, stabilized_color, 0.5, 0)
            cv2.imshow("Stabilized Frame", stabilized_frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

