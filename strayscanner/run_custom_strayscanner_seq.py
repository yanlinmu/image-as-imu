import torch
import roma
import hydra
from pathlib import Path
from omegaconf import DictConfig
from torchvision.transforms.functional import normalize
from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.funcs import disambiguate_direction
from iaai.utils.viz import viz_time_series
from iaai.utils.io import read_img_path, load_strayscanner_data, load_ckpt

@hydra.main(config_path="./configs", config_name="run_strayscanner_cfg", version_base="1.3")
def main(cfg: DictConfig):
    #
    print("cfg.save_vels =", cfg.save_vels)
    """Evaluate blur model."""
    seq_dir = Path(cfg.data_path)
    
    # Load model
    checkpoint = load_ckpt(cfg.ckpt_path) # 加载权重文件
    model = Blur2PoseSegNeXtBackbone(supervise_pose=True) # 实例化网络模型，参数随机
    model.load_state_dict(checkpoint, strict=False) # checkpoint权重赋值给模型
    model.eval().cuda() # 设置模型为评估模式，移到GPU上运行
    
    # Load data
    data = load_strayscanner_data(seq_dir)
    rgb_files = data['rgb_files']
    frame_ts = data['frame_ts']
    imu_ts = data['imu_ts']
    arkit_ts = data['arkit_ts']
    fx, fy = data['fx'], data['fy']
    imu_data = data['imu_data']
    vel_data = data['vel_data']
    
    # Setup output directory
    suffix = "" if cfg.save_suffix is None else "_" + cfg.save_suffix
    precomputed_vels_dir = seq_dir / f"precomputed_vels_iaai{suffix}"
    precomputed_vels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if precomputed velocities exist
    if cfg.precomputed_vels_path:
        pred_trans_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_trans_vels.pt")
        pred_rot_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_rot_vels.pt")
    else:
        pred_trans_vels = []
        pred_rot_vels = []
        aRb_preds = []
        atb_preds = []
        
        for k, rgb_file in enumerate(rgb_files):
            rgb_img = read_img_path(rgb_file) # 读取图像
            rgb_norm = normalize(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 图像标准化
            
            with torch.no_grad(): # 关闭梯度计算，因为这里是推理，不需要训练
                data = {
                    "image": rgb_norm, # 标准化后的图像
                    "fl": torch.tensor([0.5*(fx+fy)]).float() # 相机焦距的平均值
                }
                out = model(data) # data传入模型得到输出字典out
            flow_HW2 = out["flow_field"].squeeze().permute(1, 2, 0) # 从模型输出中提取光流
            depth_HW = out["depth"].squeeze() # 提取深度图
            pred_pose_B6 = out["pose"].squeeze(-1) # 提取预测的位姿
            residual = out["residual"].squeeze() # 提取输出的残差
            aRb_pred = roma.euler_to_rotmat('XYZ', pred_pose_B6[:, :3]).squeeze().cpu() # 欧拉角转换为旋转矩阵，tensor移回CPU
            atb_pred = pred_pose_B6[:, 3:].squeeze().cpu() # 提取预测的平移向量，tensor移回CPU

            # Disambiguate motion direction 计算每帧的旋转和平移速度
            if k < len(rgb_files) - 1: # 只对非最后一帧进行速度计算，因为速度计算需要相邻两帧的图像和时间差
                # Transform the predicted rotation to match the gyroscope's reference frame
                transform = torch.tensor([[ 0, -1,  0],
                                          [-1,  0,  0],
                                          [ 0,  0, -1]]).float() # 坐标系转换到陀螺仪参考系
                rot_vel, trans_vel = disambiguate_direction(
                    flow_HW2.cpu(),
                    0.01, # StrayScanner currently doesn't provide exposure, but can assume it's about 10ms
                    (frame_ts[k+1] - frame_ts[k]),
                    aRb_pred.float(),
                    atb_pred.float(),
                    rgb_img.cpu().squeeze(),
                    read_img_path(rgb_files[k+1], device='cpu').squeeze(),
                    read_img_path(rgb_files[k-1], device='cpu').squeeze() if k > 0 else None,
                    transform,
                )
                pred_trans_vels.append(trans_vel)
                pred_rot_vels.append(rot_vel)
                aRb_preds.append(roma.euler_to_rotmat('XYZ', rot_vel * (frame_ts[k+1] - frame_ts[k]))) # 速度积分得到位姿增量，用于后续处理如可视化
                atb_preds.append(trans_vel * (frame_ts[k+1] - frame_ts[k])) # 速度积分得到位姿增量，用于后续处理如可视化
        # 将循环中累积的列表转换成pytorch tensor，方便后续处理（列表无法直接在GPU上做矩阵运算）
        pred_trans_vels = torch.stack(pred_trans_vels)
        pred_rot_vels = torch.stack(pred_rot_vels)
        
        if cfg.save_vels:
            torch.save(pred_trans_vels, precomputed_vels_dir / "pred_trans_vels.pt")
            torch.save(pred_rot_vels, precomputed_vels_dir / "pred_rot_vels.pt")
    
    # Evaluate against ground truth
    imu_gyro_N3 = imu_data[['alpha_x', 'alpha_y', 'alpha_z']].values
    arkit_tvel_N3 = vel_data[['t_x', 't_y', 't_z']].values
    
    pred_trans_vels = pred_trans_vels.cpu().numpy()
    pred_rot_vels = pred_rot_vels.cpu().numpy()
    
    if cfg.viz_metrics:
        # Remove last frame for blur model 
        frame_ts = frame_ts[:-1]
        title_prefix = "IaaI"
        rot_title = title_prefix + " - Angular Vels vs Gyroscope"
        trans_title = title_prefix + " - Translational Vels vs ARKit"
        
        viz_time_series(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts, 
                       rot_or_trans="rot", title=rot_title)
        viz_time_series(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts, 
                       rot_or_trans="trans", title=trans_title)

if __name__ == "__main__":
    main() 