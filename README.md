# cummin 算子（Triton-Ascend 实现）

## 概述

基于 Triton-Ascend 实现的 `torch.cummin` 等价算子，返回 `(values, indices)`。
精度达到与 PyTorch（CPU/NPU 一致行为）**位级一致**（二进制一致），综合性能加速比 **1.60x**（几何平均，要求 ≥0.6x）。

## 目录结构

```
cummin/
├── cummin_task.py                  # 任务定义：参考 Model（torch.cummin）+ 测试输入组（28 组）
├── cummin_triton_ascend_impl.py    # Triton-Ascend 实现（ModelNew）
└── verify/
    ├── cummin_torch.py             # 任务文件副本（验证输入）
    ├── cummin_triton_ascend_impl.py
    ├── verify_result.json          # 精度验证结果：28/28 通过
    ├── perf_result.json            # 性能结果：几何平均 1.6012x
    ├── verify_run.log / bench_run.log
    └── verify_run2.log / bench_run2.log
```

## 精度语义（关键推导）

通过对 PyTorch CPU/NPU 的黑盒对拍，归纳出 cummin 的精确结合规则：

```
combine(run, x) = (isnan(x) || x <= run) ? x : run     # 值与索引同步更新
```

- **tie（值相等）**：更新为后出现者（last-occurrence），非保留先者；
- **±0.0**：`+0 <= -0` 与 `-0 <= +0` 均成立（IEEE 相等）→ 互相更新（与 PyTorch 实测一致）；
- **NaN**：`isnan(x)` 优先夺取 running，一旦出现永久传播，索引固定为 NaN 首现位置；
- **NaN 位模式**：PyTorch 按位保持传播输入 NaN（含非 canonical NaN），实现采用 where 选值（位保持），running 侧经标量级规范化（消除 reduce 的 0x7fffffff 污染）；
- 该规则满足结合律，可直接用于并行 scan 分解。

## 实现架构

### 核心 kernel

| Kernel | 作用 |
|--------|------|
| `cummin_packed_kernel` | 行打包路径（T==1）：多行合并为 [R, N] tile，单元组 associatve_scan(axis=1) 做**值 scan**，再用 `(x==p) \| isnan` 重构候选位置、cummax scan 得 **last-occurrence 索引** |
| `cummin_seg_scan_kernel` | 分段路径（T>1）K1：每 (row, seg) 程序独立段内双 scan（无 running 注入），输出段内 scan 结果与段尾 (块 min, 块 argmin) |
| `cummin_block_scan_kernel` | K2：每行对块 min 序列做 shifted 前缀 scan（exclusive running）；索引通道采用 **rank-cummax** 方案（valid 标志 → cumsum rank → (rank<<20\|bi) 的 max scan），绕开元组 scan 的 tie 索引 bug |
| `cummin_apply_kernel` | K3：每段 elementwise 合并 running（`take = isnan(p1) || p1 <= rv`），利用结合性免重 scan |
| `cummin_transpose_kernel` | tile 转置（dim=0 / 中间维路径：转置 → 行 scan → 转置回） |

### 调度策略（host 侧）

- `dim == last`（含 1D）：行 scan 路径；
  - `N ≤ B`（B=4096，i64 为 2048）：**行打包**单 kernel（R = min(rows, 4096/N_pad)）；
  - 否则：3-kernel（seg scan → block scan → apply），段级并行 grid = rows×T；
- `dim == 0`（2D）：物理转置 → 行 scan → 转置回；
- 其他 dim（N 维中间维）：permute 使 scan 维为第 0 维后走 dim=0 路径，输出逆变换。

### dtype 支持

float32 / float16 / bfloat16（scan 在 f32 域进行，store 时 cast 回，值域精确）；
int32 / int64（int 域 scan，NEG 用 IMAX/ILMAX）。
indices 输出统一 int64（kernel 内 int32，store 时扩展）。

## 已解决的平台问题（调试过程中发现）

1. **元组 associatve_scan 的 tie 索引 bug**：值通道正确、索引通道在跨步传播时丢失更新（2D 必现、1D tie 触发）→ 全面改用**双单元组 scan**（值 min-scan + 候选位置 cummax 重构）；
2. **循环内多次 scan 调用结果异常**（f32 大 T 段循环从第 2 段起错）→ 3-kernel 结构消除 kernel 内多段 scan 循环；
3. **浮点 reduce 的 NaN 位污染**（sum/min 产生 0x7fffffff 非 canonical）→ 提取处用标量级 where 规范化；
4. **UB 溢出**（64×64 tile 双 scan + 循环 multi-buffer 放大）→ tile 尺寸按 dtype 调整（f32 64×64 / i64 32×64）；
5. **2D 转置回写 grid 半覆盖**（f32 时 BM=BN 掩盖，i64 时暴露）→ grid 公式统一按 kernel 实际收到的 BN 计算。

## 性能数据（vNPU 181 / Ascend 910B2，warmup=3, repeats=20）

| 场景 | PyTorch | 本实现 | 加速比 |
|------|---------|--------|--------|
| 1D 1M f32 | 739.7ms | 4.2ms | **176x** |
| 1D 8192 f32（torch 慢路径）| 5.9ms | 0.26ms | **23x** |
| 2D 2048² i64 dim=-1 | 17.2ms | 15.3ms | 1.13x |
| 2D 512×512 f32 dim=-1 | 0.56ms | 0.75ms | 0.75x |
| **几何平均（28 case）** | — | — | **1.6012x** |

已知受限场景：4096² f32/f16 dim=-1（0.13-0.19x）——平台 `associative_scan` 实现的每元素固定成本（~26ns）远高于 CANN 原生 kernel 的串行向量化传播（~0.5ns），属 Triton-Ascend 编译器层面的实现差距；微小 case（torch host 短路 <0.05ms）受 kernel launch 固定开销限制。

## 复现

```bash
cd verify
python3 ../.verify_scripts/verify.py --op_name cummin --verify_dir . --triton_impl_name triton_ascend_impl
python3 ../.verify_scripts/benchmark.py --op_name cummin --verify_dir . --triton_impl_name triton_ascend_impl --warmup 3 --repeats 20 --output perf_result.json
```
