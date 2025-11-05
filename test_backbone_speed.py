import torch
import time
from iaai.modules import MSCAN

def measure_inference(model, variant_name, device="cuda", runs=30):
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 224, 320).to(device)
    data = {"image": x}

    # 预热几次
    for _ in range(5):
        with torch.no_grad():
            _ = model(data)

    # 计时
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(data)
    torch.cuda.synchronize()
    end = time.time()

    # 计算平均时间
    avg_time = (end - start) / runs * 1000  # 毫秒
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    # 显存占用（MB）
    mem_used = torch.cuda.max_memory_allocated(device=device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)

    print("=" * 60)
    print(f"Backbone variant: {variant_name}")
    print(f"Total parameters: {total_params:.2f} M")
    print(f"Average inference time: {avg_time:.2f} ms per image")
    print(f"Peak GPU memory usage: {mem_used:.2f} MB")
    print("=" * 60)

# ======================== 主程序 ============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试 Tiny 版本
    model_tiny = MSCAN(variant="tiny")
    measure_inference(model_tiny, "MSCAN-Tiny", device)

    # 测试 Base 版本
    model_base = MSCAN(variant="base")
    measure_inference(model_base, "MSCAN-Base", device)
