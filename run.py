import torch
from PIL import Image
from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.funcs import compute_single_image_velocity
import time
from safetensors.torch import load_file

model = Blur2PoseSegNeXtBackbone(device="cuda")
# state_dict = torch.load("/home/myl/image-as-an-imu/checkpoints/segnext_iaai.pth")
state_dict = load_file("/home/myl/image-as-an-imu/checkpoints/mscan_tiny/stage2/1018_135236/ckpt_000019/model.safetensors")
model.load_state_dict(state_dict, strict=True)
model.eval().cuda()

# gyro坐标系转到相机坐标系下的旋转矩阵
opencv2gyro_tf = torch.tensor([[ 0, -1,  0],
                               [-1,  0,  0],
                               [ 0,  0, -1]]).float()
# gyro坐标系角速度
omega_gyro = torch.tensor([[6.73, -4.47, 3.07]], dtype=torch.float32)
# omega_gyro = torch.tensor([[0, 0, 0]], dtype=torch.float32)
# 转换到相机坐标系
omega_cam = omega_gyro @ opencv2gyro_tf.T

demo_img = Image.open("./assets/demo.jpg")
data = {"image": demo_img, "fx": 251.3630, "fy": 251.7480, "imu_gyro": omega_cam} # fx, fy相机内参中的焦距
# 单张图片推理计时
torch.cuda.synchronize()
t0 = time.time()
out = model.infer(data) # out is a dict with keys "flow_field", "depth", "pose", "residual"
torch.cuda.synchronize()
t1 = time.time()
fps = 1.0 / (t1 - t0)
print(f"Inference time: {t1 - t0:.4f} s, FPS: {fps:.2f}")  # 单帧耗时

# Optionally apply a transformation to the rotation matrix
# This transforms the rotation to match the iPhone gyroscope's coordinate system
# opencv2gyro_tf = torch.tensor([[ 0, -1,  0],
#                                [-1,  0,  0],
#                                [ 0,  0, -1]]).float()
# rot_vel, trans_vel = compute_single_image_velocity(
#     out["pose"].squeeze(),
#     exposure_time_s=0.01,
#     M=opencv2gyro_tf,
#     device="cuda")
trans_vel = compute_single_image_velocity(
    out["pose"].squeeze(),
    exposure_time_s=0.01,
    M=opencv2gyro_tf,
    device="cuda")

# Note that the velocity has sign ambiguity if there is only one image
# print(f"Rotational velocity: {rot_vel.cpu().numpy()}")
print(f"Translational velocity: {trans_vel.cpu().numpy()}")