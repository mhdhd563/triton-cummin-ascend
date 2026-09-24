import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        return torch.cummin(x, dim=dim)


def _inject_nan(x, step=997):
    flat = x.flatten()
    pos = torch.arange(0, flat.numel(), step, device=x.device)
    nan_t = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    flat.scatter_(0, pos, nan_t.expand(pos.numel()))
    return x.reshape(x.shape)


def _inject_signed_zero(x):
    flat = x.flatten()
    n = flat.numel()
    bits = flat.view(torch.int32 if x.dtype == torch.float32 else torch.int16)
    pm = torch.arange(0, n, 31, device=x.device)
    nm = torch.arange(3, n, 47, device=x.device)
    pzero = torch.zeros(pm.numel(), dtype=bits.dtype, device=x.device)
    nzero = torch.full((nm.numel(),), -0x80000000 if x.dtype == torch.float32 else -0x8000,
                       dtype=bits.dtype, device=x.device)
    bits.scatter_(0, pm % n, pzero)
    bits.scatter_(0, nm % n, nzero)
    return bits.view(x.dtype).reshape(x.shape)


def get_input_groups():
    groups = []
    dev = 'npu'
    g = torch.Generator(device='cpu').manual_seed(42)

    # --- 精度覆盖（小规模） ---
    groups.append((torch.randn(8192, generator=g).to(dev), 0))
    for dt in (torch.float16, torch.bfloat16, torch.int32, torch.int64):
        if dt in (torch.int32, torch.int64):
            x = torch.randint(-100000, 100000, (8192,), generator=g).to(dev).to(dt)
        else:
            x = torch.randn(8192, generator=g).to(dev).to(dt)
        groups.append((x, 0))

    # tie 密集
    groups.append((torch.randint(0, 8, (8192,), generator=g).to(dev), 0))
    # NaN
    groups.append((_inject_nan(torch.randn(8192, generator=g).to(dev)), 0))
    # signed zero
    groups.append((_inject_signed_zero(torch.randn(8192, generator=g).to(dev)), 0))
    # 边界: 长度1 / 递减 / 全等
    groups.append((torch.randn(1, generator=g).to(dev), 0))
    groups.append((torch.arange(8192, 0, -1, dtype=torch.float32).to(dev), 0))
    groups.append((torch.full((8192,), 3.14159, device=dev), 0))

    # --- 2D ---
    groups.append((torch.randn(512, 512, generator=g).to(dev), -1))
    groups.append((torch.randn(512, 512, generator=g).to(dev), 0))
    groups.append((torch.randn(256, 512, generator=g).to(dev).to(torch.float16), -1))
    groups.append((torch.randn(256, 512, generator=g).to(dev).to(torch.bfloat16), 0))
    groups.append((torch.randint(-100000, 100000, (256, 512), generator=g).to(dev).to(torch.int32), -1))
    groups.append((torch.randint(-100000, 100000, (256, 512), generator=g).to(dev).to(torch.int64), 0))
    # tie 2D
    groups.append((torch.randint(0, 8, (256, 512), generator=g).to(dev), -1))
    # NaN 2D
    groups.append((_inject_nan(torch.randn(256, 512, generator=g).to(dev)), 0))

    # --- 3D 中间维 ---
    groups.append((torch.randn(8, 16, 128, generator=g).to(dev), 1))
    groups.append((torch.randn(8, 16, 128, generator=g).to(dev), 0))
    groups.append((torch.randn(8, 16, 128, generator=g).to(dev), 2))

    # --- 性能场景（大规模） ---
    groups.append((torch.randn(1 << 20, generator=g).to(dev), 0))
    groups.append((torch.randint(-100000, 100000, (1 << 20,), generator=g).to(dev).to(torch.int64), 0))
    groups.append((torch.randn(4096, 4096, generator=g).to(dev), -1))
    groups.append((torch.randn(4096, 4096, generator=g).to(dev), 0))
    groups.append((torch.randn(4096, 4096, generator=g).to(dev).to(torch.float16), -1))
    groups.append((torch.randint(-100000, 100000, (2048, 2048), generator=g).to(dev).to(torch.int64), -1))
    return groups


def get_inputs():
    return get_input_groups()[0]


def get_init_inputs():
    return []
