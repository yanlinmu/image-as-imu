import torch
from iaai.modules import MSCAN

# 随机生成一张输入图像 [B, C, H, W]
x = torch.randn(1, 3, 224, 320)

# 初始化 tiny 版本的 SegNeXt backbone
backbone = MSCAN(variant="tiny")

# 前向推理（注意这里是字典输入）
out = backbone({"image": x})

# 打印输出信息
print("=" * 60)
print("Output keys:", out.keys())

# 逐层打印 feature shape
features = out["features"]
print(f"Number of feature maps: {len(features)}")
for i, f in enumerate(features):
    print(f" Stage {i+1} feature shape: {tuple(f.shape)}")

# 参数量统计
total_params = sum(p.numel() for p in backbone.parameters()) / 1e6
print(f"Total parameters: {total_params:.2f} M")
print("=" * 60)
