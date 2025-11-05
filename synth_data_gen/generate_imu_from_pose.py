import pandas as pd
import numpy as np
import torch
import cv2
import roma  # pip install roma

def compute_angular_velocity_from_quat(qx, qy, qz, qw, dt):
    """
    Compute angular velocity vector (rad/s) from a rotation quaternion over Δt.
    qx, qy, qz, qw define rotation R_b<-a.
    """
    # 四元数转旋转矩阵
    q = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    R = roma.unitquat_to_rotmat(q[None]).squeeze()  # 3x3
    R_np = R.cpu().numpy()

    # 转axis-angle向量
    rot_vec, _ = cv2.Rodrigues(R_np)  # shape (3, 1)
    rot_vec = torch.from_numpy(rot_vec).squeeze()

    # # 除以时间间隔得到角速度
    # omega = rot_vec / dt
    return rot_vec


def main(csv_path, save_path=None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    imu_gyros = []
    for i, row in df.iterrows():
        try:
            dt = row["ts2"] - row["ts1"]
            if dt <= 0:
                imu_gyros.append([np.nan, np.nan, np.nan])
                continue

            omega = compute_angular_velocity_from_quat(
                row["bRa_qx"], row["bRa_qy"], row["bRa_qz"], row["bRa_qw"], dt
            )
            imu_gyros.append(omega.numpy())
        except Exception as e:
            print(f"Error at row {i}: {e}")
            imu_gyros.append([np.nan, np.nan, np.nan])

    imu_gyros = np.vstack(imu_gyros)
    df["imu_gyro_x"] = imu_gyros[:, 0]
    df["imu_gyro_y"] = imu_gyros[:, 1]
    df["imu_gyro_z"] = imu_gyros[:, 2]

    if save_path is None:
        save_path = csv_path.replace(".csv", "_with_imu_rad_per_frame.csv")

    df.to_csv(save_path, index=False)
    print(f"Saved augmented CSV to {save_path}")
    print(df[["imu_gyro_x", "imu_gyro_y", "imu_gyro_z"]].head())


if __name__ == "__main__":
    # 修改为你自己的CSV路径
    csv_path = "/home/myl/image-as-an-imu/synth_data_gen/synth_data_train-skip_10.csv"
    # csv_path = "/home/myl/image-as-an-imu/synth_data_gen/synth_data_val-skip_10.csv"
    main(csv_path)
