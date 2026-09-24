import torch
import torch.nn as nn
import triton
import triton.language as tl

BI_BITS = 20
BI_MASK = (1 << BI_BITS) - 1


@triton.jit
def _cmin(a, b):
    return tl.where((b != b) | (b <= a), b, a)


@triton.jit
def _cmax(a, b):
    return tl.where(b > a, b, a)


@triton.jit
def cummin_seg_scan_kernel(x_ptr, p_ptr, g_ptr, bv_ptr, bi_ptr, N, T,
                           IS_HALF: tl.constexpr, IS_FLOAT: tl.constexpr,
                           NEG: tl.constexpr, B: tl.constexpr, INLINE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // T
    seg = pid % T
    base_in = row.to(tl.int64) * N + seg * B
    offs = tl.arange(0, B)
    o = seg * B + offs
    m = o < N
    x = tl.load(x_ptr + base_in + offs, mask=m, other=NEG)
    if IS_HALF:
        x = x.to(tl.float32)
    IMIN = -2147483647 - 1
    p = tl.associative_scan(x, 0, _cmin)
    y = tl.where((x == p) | ((p != p) & (x != x)), o.to(tl.int32), IMIN)
    idx = tl.associative_scan(y, 0, _cmax)
    last_pos = tl.max(tl.where(m, offs, -1), axis=0)
    tail_v = tl.sum(tl.where(offs == last_pos, p, 0), axis=0)
    tail_i = tl.sum(tl.where(offs == last_pos, idx, 0), axis=0)
    if IS_FLOAT:
        tail_v = tl.where(tail_v != tail_v, float('nan'), tail_v)
    out_off = row * T + seg
    tl.store(bv_ptr + out_off, tail_v)
    tl.store(bi_ptr + out_off, tail_i)
    if INLINE:
        if IS_HALF:
            tl.store(p_ptr + base_in + offs, p.to(p_ptr.dtype.element_ty), mask=m)
        else:
            tl.store(p_ptr + base_in + offs, p, mask=m)
        tl.store(g_ptr + base_in + offs, idx.to(tl.int64), mask=m)
    else:
        tl.store(p_ptr + base_in + offs, p, mask=m)
        tl.store(g_ptr + base_in + offs, idx.to(tl.int64), mask=m)


@triton.jit
def cummin_block_scan_kernel(bv_ptr, bi_ptr, rv_ptr, ri_ptr, T,
                             IS_FLOAT: tl.constexpr, NEG: tl.constexpr, TB: tl.constexpr):
    row = tl.program_id(0)
    t = tl.arange(0, TB)
    m = t < T
    has_prev = (t > 0) & m
    prev = tl.where(t > 0, t - 1, 0)
    bv_prev = tl.load(bv_ptr + row * T + prev, mask=has_prev, other=NEG)
    bi_prev = tl.load(bi_ptr + row * T + prev, mask=has_prev, other=0)
    prev_min = tl.associative_scan(bv_prev, 0, _cmin)
    if IS_FLOAT:
        valid = (bv_prev != bv_prev) | (bv_prev <= prev_min)
    else:
        valid = bv_prev <= prev_min
    valid = tl.where(has_prev, valid, False)
    valid = tl.where(m & has_prev, valid, False)
    rank = tl.cumsum(valid.to(tl.int32), 0)
    key = tl.where(valid, (rank.to(tl.int64) << 20) | (bi_prev & 0xFFFFF), 0)
    key = tl.where(has_prev, key, 0)
    key_max = tl.associative_scan(key, 0, _cmax)
    rv = prev_min
    ri = (key_max & 0xFFFFF).to(tl.int32)
    tl.store(rv_ptr + row * T + t, rv, mask=m)
    tl.store(ri_ptr + row * T + t, ri, mask=m)


@triton.jit
def cummin_apply_kernel(p_ptr, g_ptr, rv_ptr, ri_ptr, v_ptr, i_ptr, N, T,
                        IS_HALF: tl.constexpr, IS_FLOAT: tl.constexpr, NEG: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // T
    seg = pid % T
    base_in = row.to(tl.int64) * N + seg * B
    offs = tl.arange(0, B)
    o = seg * B + offs
    m = o < N
    p1 = tl.load(p_ptr + base_in + offs, mask=m, other=NEG)
    g1 = tl.load(g_ptr + base_in + offs, mask=m, other=0)
    rv = tl.load(rv_ptr + row * T + seg)
    ri = tl.load(ri_ptr + row * T + seg)
    if IS_FLOAT:
        take = (p1 != p1) | (p1 <= rv)
    else:
        take = p1 <= rv
    take = tl.where(m, take, False)
    fv = tl.where(take, p1, rv)
    fi = tl.where(take, g1, ri.to(tl.int64))
    if IS_HALF:
        tl.store(v_ptr + base_in + offs, fv.to(v_ptr.dtype.element_ty), mask=m)
    else:
        tl.store(v_ptr + base_in + offs, fv, mask=m)
    tl.store(i_ptr + base_in + offs, fi.to(tl.int64), mask=m)


@triton.jit
def cummin_packed_kernel(x_ptr, v_ptr, i_ptr, rows, N,
                         IS_HALF: tl.constexpr, IS_FLOAT: tl.constexpr, NEG: tl.constexpr,
                         R: tl.constexpr, NP: tl.constexpr):
    pid = tl.program_id(0)
    rr = pid * R + tl.arange(0, R)
    nn = tl.arange(0, NP)
    m = (rr[:, None] < rows) & (nn[None, :] < N)
    base = rr[:, None].to(tl.int64) * N + nn[None, :]
    x = tl.load(x_ptr + base, mask=m, other=NEG)
    if IS_HALF:
        x = x.to(tl.float32)
    IMIN = -2147483647 - 1
    p = tl.associative_scan(x, 1, _cmin)
    y = tl.where((x == p) | ((p != p) & (x != x)), nn[None, :].to(tl.int32), IMIN)
    y = tl.where(m, y, IMIN)
    idx = tl.associative_scan(y, 1, _cmax)
    if IS_HALF:
        tl.store(v_ptr + base, p.to(v_ptr.dtype.element_ty), mask=m)
    else:
        tl.store(v_ptr + base, p, mask=m)
    tl.store(i_ptr + base, idx.to(tl.int64), mask=m)


@triton.jit
def cummin_transpose_kernel(x_ptr, o_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pn = pid * BN + tl.arange(0, BN)
    ncols = tl.arange(0, BM)
    for m0 in range(0, M, BM):
        pm = m0 + ncols
        m = (pm[:, None] < M) & (pn[None, :] < N)
        offs_in = pm[:, None] * N + pn[None, :]
        x = tl.load(x_ptr + offs_in, mask=m, other=0)
        xt = tl.trans(x)
        offs_out = pn[:, None] * M + pm[None, :]
        mo = (pn[:, None] < N) & (pm[None, :] < M)
        tl.store(o_ptr + offs_out, xt, mask=mo)


def _dtype_cfg(dt):
    if dt == torch.float32:
        return False, True, float('inf'), 4096
    if dt in (torch.float16, torch.bfloat16):
        return True, True, float('inf'), 4096
    if dt == torch.int32:
        return False, False, 2147483647, 4096
    return False, False, 9223372036854775807, 2048


def _row_scan(x, values, indices):
    rows = x.numel() // x.shape[-1]
    N = x.shape[-1]
    dt = x.dtype
    is_half, is_float, neg, B = _dtype_cfg(dt)
    T = (N + B - 1) // B
    total = rows * T
    p_dtype = torch.float32 if is_float else dt
    bv = torch.empty(rows, T, device=x.device, dtype=p_dtype)
    bi = torch.empty(rows, T, device=x.device, dtype=torch.int32)
    if T > 1:
        p1 = torch.empty(rows, N, device=x.device, dtype=p_dtype)
        g1 = torch.empty(rows, N, device=x.device, dtype=torch.int64)
        rv = torch.empty(rows, T, device=x.device, dtype=p_dtype)
        ri = torch.empty(rows, T, device=x.device, dtype=torch.int32)
        cummin_seg_scan_kernel[(total,)](x, p1, g1, bv, bi, N, T,
                                         IS_HALF=is_half, IS_FLOAT=is_float, NEG=neg, B=B, INLINE=False)
        TB = triton.next_power_of_2(max(T, 2))
        if TB > 4096:
            raise RuntimeError("row length exceeds supported scan depth")
        cummin_block_scan_kernel[(rows,)](bv, bi, rv, ri, T,
                                          IS_FLOAT=is_float, NEG=neg, TB=TB)
        cummin_apply_kernel[(total,)](p1, g1, rv, ri, values, indices, N, T,
                                      IS_HALF=is_half, IS_FLOAT=is_float, NEG=neg, B=B)
    else:
        NP = triton.next_power_of_2(max(N, 2))
        MT = 2048 if dt == torch.int64 else 4096
        R = max(1, min(rows, MT // NP))
        cummin_packed_kernel[((rows + R - 1) // R,)](x, values, indices, rows, N,
                                                     IS_HALF=is_half, IS_FLOAT=is_float, NEG=neg,
                                                     R=R, NP=NP)


def _col_scan(x, values, indices):
    M, N = x.shape
    BM, BN = (32, 64) if x.dtype == torch.int64 else (64, 64)
    xt = torch.empty(N, M, device=x.device, dtype=x.dtype)
    cummin_transpose_kernel[((N + BN - 1) // BN,)](x, xt, M, N, BM=BM, BN=BN)
    vt = torch.empty_like(xt)
    it = torch.empty(N, M, device=x.device, dtype=torch.int64)
    _row_scan(xt, vt, it)
    cummin_transpose_kernel[((M + BM - 1) // BM,)](vt, values, N, M, BM=BN, BN=BM)
    cummin_transpose_kernel[((M + BM - 1) // BM,)](it, indices, N, M, BM=BN, BN=BM)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        if not x.is_contiguous():
            x = x.contiguous()
        ndim = x.dim()
        if dim < 0:
            dim = dim + ndim
        values = torch.empty_like(x)
        indices = torch.empty(x.shape, device=x.device, dtype=torch.int64)
        if ndim == 1 or dim == ndim - 1:
            _row_scan(x, values, indices)
        elif dim == 0 and ndim == 2:
            _col_scan(x, values, indices)
        else:
            order = [dim] + [d for d in range(ndim) if d != dim]
            xs = x.permute(*order).contiguous()
            S = xs.shape[0]
            C = xs.numel() // S
            xs2 = xs.reshape(S, C)
            BM, BN = (32, 64) if xs2.dtype == torch.int64 else (64, 64)
            xt = torch.empty(C, S, device=x.device, dtype=xs2.dtype)
            cummin_transpose_kernel[((C + BN - 1) // BN,)](xs2, xt, S, C, BM=BM, BN=BN)
            vt = torch.empty_like(xt)
            it = torch.empty(C, S, device=x.device, dtype=torch.int64)
            _row_scan(xt, vt, it)
            vs2 = torch.empty_like(xs2)
            iss2 = torch.empty(S, C, device=x.device, dtype=torch.int64)
            cummin_transpose_kernel[((S + BM - 1) // BM,)](vt, vs2, C, S, BM=BN, BN=BM)
            cummin_transpose_kernel[((S + BM - 1) // BM,)](it, iss2, C, S, BM=BN, BN=BM)
            inv = [order.index(d) for d in range(ndim)]
            shape_p = tuple(x.shape[d] for d in order)
            values.copy_(vs2.reshape(shape_p).permute(*inv).contiguous())
            indices.copy_(iss2.reshape(shape_p).permute(*inv).contiguous())
        return values, indices
