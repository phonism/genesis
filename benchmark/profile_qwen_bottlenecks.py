"""
Profile Qwen forward/backward to identify performance bottlenecks.

Features
- End-to-end timing for forward/backward with warmup + repeated iters
- Per-stage CUDA Event timers: Embedding, per-layer total, Final Norm, LM Head
- Optional PyTorch Profiler (CPU+CUDA) with export to Chrome trace
- Contiguous-copy counter: counts non-contiguous -> contiguous copies and total bytes
- Memory summary from Genesis CUDA allocator (optional)

Usage
  python benchmark/profile_qwen_bottlenecks.py \
    --batch 4 --seqlen 1024 --layers 2 --iters 10 --warmup 5 --device cuda \
    --export-trace genesis_qwen_trace.json --show-memory

Notes
- Uses lightweight instrumentation that does not modify core modules
- For deeper breakdown (attention vs MLP) enable PyTorch profiler export and
  inspect the Chrome trace, or extend with module-level monkeypatch as needed
"""
import argparse
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

import genesis
from genesis.models.qwen import ModelArgs, QwenModel


class CudaTimer:
    def __init__(self):
        self.stats_ms: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)

    @contextmanager
    def region(self, name: str):
        s, e = _cuda_events()
        torch.cuda.synchronize()
        s.record()
        yield
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        self.stats_ms[name] += ms
        self.counts[name] += 1

    def reset(self) -> None:
        self.stats_ms.clear()
        self.counts.clear()


class ContiguousStats:
    calls = 0
    bytes = 0
    sources = {}  # Track where contiguous calls come from

    @classmethod
    def reset(cls):
        cls.calls = 0
        cls.bytes = 0
        cls.sources = {}


@dataclass
class StageStat:
    name: str
    total_ms: float
    count: int
    per_call_ms: float
    per_iter_ms: float


def compute_time_stats(samples: Iterable[float]) -> Dict[str, Any]:
    values = list(samples)
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    filtered = arr[arr <= threshold]
    outliers = arr[arr > threshold]

    stats: Dict[str, Any] = {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'outlier_threshold': float(threshold),
        'outliers': [float(x) for x in outliers],
        'filtered_mean': float(filtered.mean()) if filtered.size else None,
        'filtered_std': float(filtered.std()) if filtered.size else None,
    }
    return stats


def print_time_stats(label: str, stats: Dict[str, Any]) -> None:
    if not stats:
        return
    mean = stats['mean']
    std = stats['std']
    cv = 100.0 * std / mean if mean else 0.0
    print(f"\n{label}:")
    print(f"  samples: {stats['count']} | mean {mean:.2f} ms ± {std:.2f} ms (cv {cv:.1f}%)")
    print(f"  median : {stats['median']:.2f} ms | min {stats['min']:.2f} | max {stats['max']:.2f}")
    outliers = stats.get('outliers') or []
    if outliers:
        formatted = ', '.join(f"{v:.1f}" for v in outliers)
        print(f"  outliers: {len(outliers)} > {stats['outlier_threshold']:.1f} ms -> [{formatted}]")
        filtered_mean = stats.get('filtered_mean')
        filtered_std = stats.get('filtered_std')
        if filtered_mean is not None and filtered_std is not None:
            print(f"  filtered mean ± std: {filtered_mean:.2f} ms ± {filtered_std:.2f} ms")


def collect_stage_stats(timer: CudaTimer, measured_iters: int) -> List[StageStat]:
    if not timer.stats_ms:
        return []
    denom = max(1, measured_iters)
    stats: List[StageStat] = []
    for name, total in timer.stats_ms.items():
        count = timer.counts.get(name, 0)
        per_call = total / max(1, count)
        per_iter = total / denom
        stats.append(StageStat(name=name, total_ms=total, count=count, per_call_ms=per_call, per_iter_ms=per_iter))
    stats.sort(key=lambda s: s.per_iter_ms, reverse=True)
    return stats


def print_stage_stats(stage_stats: List[StageStat], fw_mean: float) -> None:
    if not stage_stats:
        return
    total_per_iter = sum(stat.per_iter_ms for stat in stage_stats)
    print("\nStage Breakdown (avg per iteration):")
    for stat in stage_stats:
        share_fw = 100.0 * stat.per_iter_ms / fw_mean if fw_mean else 0.0
        print(
            f"  {stat.name:20s} {stat.per_iter_ms:7.2f} ms/iter  "
            f"({share_fw:5.1f}% fw, {stat.count:3d} calls, {stat.per_call_ms:6.2f} ms/call)"
        )
    print(f"  Total recorded stage time: {total_per_iter:.2f} ms/iter")


def patch_contiguous_counter():
    """Wrap CUDAStorage.contiguous to count non-contiguous copies and bytes moved."""
    import traceback
    try:
        from genesis.backends.cuda import CUDAStorage
    except Exception:
        return None

    orig = CUDAStorage.contiguous

    def wrapped(self):
        if hasattr(self, 'is_contiguous') and not self.is_contiguous():
            ContiguousStats.calls += 1
            try:
                ContiguousStats.bytes += int(self.size) * int(self.itemsize)
                # Capture call stack to identify source
                stack = traceback.extract_stack()
                for frame in reversed(stack[:-1]):  # Skip this frame
                    # Look for meaningful operation names
                    if 'matmul' in frame.filename or 'reshape' in frame.filename or \
                       'transpose' in frame.filename or 'ops' in frame.filename:
                        key = f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
                        ContiguousStats.sources[key] = ContiguousStats.sources.get(key, 0) + 1
                        break
                else:
                    # If no specific op found, record the immediate caller
                    if len(stack) > 2:
                        frame = stack[-2]
                        key = f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}"
                        ContiguousStats.sources[key] = ContiguousStats.sources.get(key, 0) + 1
            except Exception:
                pass
        return orig(self)

    setattr(CUDAStorage, 'contiguous', wrapped)
    return orig


class ProfiledQwen(QwenModel):
    """QwenModel with stage timers: Embedding, per-layer, Final Norm, LM Head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = CudaTimer()

    def forward(self, idx, position_ids=None):
        batch_size, sequence_length = idx.shape

        with self.timer.region('embed'):  # Embedding
            x = self.embed_tokens(idx)

        if position_ids is None:
            position_ids = genesis.arange(0, sequence_length, device=idx.device)
            position_ids = position_ids + genesis.zeros(batch_size, sequence_length, device=idx.device)
        mask = None

        # Per-layer total timing
        for i, layer in enumerate(self.layers):
            with self.timer.region(f'layer_{i:02d}'):  # layer total
                x = layer(x, position_ids, position_ids, mask)

        with self.timer.region('final_norm'):
            x = self.norm(x)
        with self.timer.region('lm_head'):
            x = self.lm_head(x)
        return x


# -------------------- Deep profiling (per-module) --------------------
class DeepStats:
    linear = {}
    embedding = {}
    sdpa_ms = 0.0
    sdpa_calls = 0
    non_fp16_linear = 0
    non_contig_inputs = 0

    @classmethod
    def reset(cls):
        cls.linear = {}
        cls.embedding = {}
        cls.sdpa_ms = 0.0
        cls.sdpa_calls = 0
        cls.non_fp16_linear = 0
        cls.non_contig_inputs = 0


# -------------------- Backward op profiling --------------------
PROFILE_BACKWARD_ACTIVE = False

class BackwardStats:
    op = {}           # op_name -> {'ms': float, 'calls': int}
    gemm = {}         # shape_key -> {'ms': float, 'calls': int, 'shapes': {...}}
    non_fp16_matmul = 0

    @classmethod
    def reset(cls):
        cls.op = {}
        cls.gemm = {}
        cls.non_fp16_matmul = 0


def _cuda_events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def install_deep_profilers(model: QwenModel):
    """Monkeypatch Linear/Embedding/SDPA to record per-module timings and shapes."""
    from genesis.nn.modules.linear import Linear
    from genesis.nn.modules.sparse import Embedding
    import genesis.nn.functional as NF

    # Wrap Linear
    def wrap_linear(mod: Linear, name: str):
        orig = mod.forward
        def fwd(x):
            try:
                if hasattr(x, 'dtype') and str(x.dtype) != str(genesis.float16):
                    DeepStats.non_fp16_linear += 1
                if hasattr(x, 'is_contiguous') and not x.is_contiguous():
                    DeepStats.non_contig_inputs += 1
            except Exception:
                pass
            s, e = _cuda_events()
            torch.cuda.synchronize(); s.record()
            y = orig(x)
            e.record(); torch.cuda.synchronize()
            ms = s.elapsed_time(e)
            ent = DeepStats.linear.get(name, {'ms':0.0,'calls':0,'shapes':None})
            ent['ms'] += ms; ent['calls'] += 1
            if ent['shapes'] is None:
                try:
                    ent['shapes'] = {'x': tuple(x.shape), 'w': tuple(mod.weight.shape)}
                except Exception:
                    ent['shapes'] = {}
            DeepStats.linear[name] = ent
            return y
        mod.forward = fwd

    # Wrap Embedding
    def wrap_embedding(mod: Embedding, name: str):
        orig = mod.forward
        def fwd(x):
            s, e = _cuda_events()
            torch.cuda.synchronize(); s.record()
            y = orig(x)
            e.record(); torch.cuda.synchronize()
            ms = s.elapsed_time(e)
            ent = DeepStats.embedding.get(name, {'ms':0.0,'calls':0})
            ent['ms'] += ms; ent['calls'] += 1
            DeepStats.embedding[name] = ent
            return y
        mod.forward = fwd

    # Wrap SDPA
    orig_sdpa = NF.scaled_dot_product_attention
    def sdpa_wrap(q, k, v, *args, **kwargs):
        s, e = _cuda_events()
        torch.cuda.synchronize(); s.record()
        out = orig_sdpa(q, k, v, *args, **kwargs)
        e.record(); torch.cuda.synchronize()
        DeepStats.sdpa_ms += s.elapsed_time(e)
        DeepStats.sdpa_calls += 1
        return out
    NF.scaled_dot_product_attention = sdpa_wrap

    # Install on Qwen modules with informative names
    wrap_embedding(model.embed_tokens, 'embed')
    wrap_linear(model.lm_head, 'lm_head')
    for li, blk in enumerate(model.layers):
        wrap_linear(blk.self_attn.q_proj, f'layer{li:02d}.q_proj')
        wrap_linear(blk.self_attn.k_proj, f'layer{li:02d}.k_proj')
        wrap_linear(blk.self_attn.v_proj, f'layer{li:02d}.v_proj')
        wrap_linear(blk.self_attn.o_proj, f'layer{li:02d}.o_proj')
        wrap_linear(blk.mlp.gate_proj,   f'layer{li:02d}.ff.gate')
        wrap_linear(blk.mlp.up_proj,     f'layer{li:02d}.ff.up')
        wrap_linear(blk.mlp.down_proj,   f'layer{li:02d}.ff.down')

    # Wrap low-level matmul to measure pure GEMM time and shapes
    # Wrap OperationDispatcher matmul for CUDA so we capture actual GEMM time
    # Generic OperationDispatcher wrappers to measure backward ops
    try:
        from genesis.ops.dispatcher import OperationDispatcher, DeviceType
        wrapped = {}
        for (op_name, dev), fn in list(OperationDispatcher._registry.items()):
            if dev != DeviceType.CUDA:
                continue
            # Skip if already wrapped
            if (op_name, dev) in wrapped:
                continue
            orig_impl = fn
            def make_wrap(op_name, orig_impl):
                def impl_wrap(*args, **kwargs):
                    global PROFILE_BACKWARD_ACTIVE
                    s, e = _cuda_events()
                    torch.cuda.synchronize(); s.record()
                    out = orig_impl(*args, **kwargs)
                    e.record(); torch.cuda.synchronize()
                    ms = s.elapsed_time(e)
                    if PROFILE_BACKWARD_ACTIVE:
                        # Aggregate by op name
                        ent = BackwardStats.op.get(op_name, {'ms':0.0,'calls':0})
                        ent['ms'] += ms; ent['calls'] += 1
                        BackwardStats.op[op_name] = ent
                        # Special handling for matmul: record shapes and dtype
                        if op_name == 'matmul' and len(args) >= 2:
                            a = args[0]; b = args[1]
                            try:
                                a_shape = tuple(getattr(a, 'shape', []))
                                b_shape = tuple(getattr(b, 'shape', []))
                                k = f"gemm_bwd {a_shape} x {b_shape}"
                                g = BackwardStats.gemm.get(k, {'ms':0.0,'calls':0,'shapes':{'a':a_shape,'b':b_shape}})
                                g['ms'] += ms; g['calls'] += 1
                                BackwardStats.gemm[k] = g
                                # dtype check
                                a_dt = getattr(a, 'dtype', None)
                                b_dt = getattr(b, 'dtype', None)
                                if str(a_dt) != str(genesis.float16) or str(b_dt) != str(genesis.float16):
                                    BackwardStats.non_fp16_matmul += 1
                            except Exception:
                                pass
                    return out
                return impl_wrap
            OperationDispatcher._registry[(op_name, dev)] = make_wrap(op_name, orig_impl)
            wrapped[(op_name, dev)] = True
    except Exception:
        pass

    return orig_sdpa


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seqlen', type=int, default=1024)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--hidden', type=int, default=896)
    p.add_argument('--heads', type=int, default=14)
    p.add_argument('--kv_heads', type=int, default=2)
    p.add_argument('--intermediate', type=int, default=4864)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--warmup', type=int, default=10, help="Warmup iterations (default: 10, increase if timings are unstable)")
    p.add_argument('--stabilize', type=int, default=10, help="Extra stabilization iterations after warmup (default: 10)")
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dtype', type=str, default='float16', choices=['float16','float32','bfloat16'])
    p.add_argument('--export-trace', type=str, default='')
    p.add_argument('--show-memory', action='store_true')
    p.add_argument('--profile-backward', action='store_true')
    p.add_argument('--deep', action='store_true', help='Enable per-module deep profiling (Linear/Embedding/SDPA)')
    return p.parse_args()


def make_model(cfg):
    args = ModelArgs(
        vocab_size=151936,
        hidden_size=cfg.hidden,
        intermediate_size=cfg.intermediate,
        n_layer=cfg.layers,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads,
        max_position_embeddings=max(cfg.seqlen, 2048),
        norm_eps=1e-6,
    )
    model = ProfiledQwen(args)
    device = genesis.device(cfg.device)
    model = model.to(device)
    return model


def make_inputs(cfg, device):
    import numpy as np
    x = genesis.tensor(np.random.randint(0, 151936, (cfg.batch, cfg.seqlen), dtype=np.int64), device=device)
    y = genesis.tensor(np.random.randint(0, 151936, (cfg.batch, cfg.seqlen), dtype=np.int64), device=device)
    return x, y


def set_dtype(model, cfg):
    if cfg.dtype == 'float16':
        for p in model.parameters():
            try:
                p.data = p.data.half()
            except Exception:
                pass
        for _, buf in model.named_buffers():
            try:
                buf = buf.half()
            except Exception:
                pass
        return
    if cfg.dtype == 'float32':
        for p in model.parameters():
            p.data = p.data.float()
    elif cfg.dtype == 'bfloat16':
        for p in model.parameters():
            p.data = p.data.to(genesis.bfloat16)


def run(cfg):
    device = genesis.device(cfg.device)
    model = make_model(cfg)
    set_dtype(model, cfg)
    x, y = make_inputs(cfg, device)

    # Loss: cross-entropy over vocab
    def loss_fn(logits, targets):
        # flatten
        B, S, V = logits.shape
        return genesis.nn.functional.cross_entropy(logits.view(B*S, V), targets.view(B*S))

    # Optional profiler
    prof_ctx = nullcontext()
    if cfg.export_trace:
        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=max(1, cfg.iters)),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.export_trace) if cfg.export_trace else None,
        )

    # Deep profiling install
    orig_sdpa = None
    if cfg.deep:
        DeepStats.reset()
        orig_sdpa = install_deep_profilers(model)

    # Patch contiguous counter
    patch_contiguous_counter()
    ContiguousStats.reset()

    # Warmup
    for _ in range(cfg.warmup):
        logits = model(x)
        if cfg.profile_backward:
            loss = loss_fn(logits, y)
            BackwardStats.reset()
            globals()['PROFILE_BACKWARD_ACTIVE'] = True
            loss.backward()
            globals()['PROFILE_BACKWARD_ACTIVE'] = False

    # Extra stabilization: run more iterations without timing to settle the GPU
    if cfg.stabilize > 0:
        print(f"Running {cfg.stabilize} extra stabilization iterations...")
        for _ in range(cfg.stabilize):
            genesis.cuda.synchronize()
            logits = model(x)
            if cfg.profile_backward:
                loss = loss_fn(logits, y)
                loss.backward()
            genesis.cuda.synchronize()

    # Reset instrumentation so the measured window excludes warmup/stabilize noise
    model.timer.reset()
    ContiguousStats.reset()
    if cfg.deep:
        DeepStats.reset()
    if cfg.profile_backward:
        BackwardStats.reset()

    print("Starting timed measurements...")
    fw_times: List[float] = []
    bw_times: List[float] = []
    loss_times: List[float] = []

    with prof_ctx:
        for _ in range(cfg.iters):
            genesis.cuda.synchronize()
            t0 = time.time()
            logits = model(x)
            genesis.cuda.synchronize()
            fw_times.append((time.time() - t0) * 1000)

            if cfg.profile_backward:
                genesis.cuda.synchronize()
                t_loss_start = time.time()
                loss = loss_fn(logits, y)
                genesis.cuda.synchronize()
                loss_time = (time.time() - t_loss_start) * 1000
                loss_times.append(loss_time)

                globals()['PROFILE_BACKWARD_ACTIVE'] = True
                t_bw_start = time.time()
                loss.backward()
                t_bw_nosync = time.time()
                genesis.cuda.synchronize()
                t_bw_sync = time.time()
                globals()['PROFILE_BACKWARD_ACTIVE'] = False

                bw_times.append((t_bw_sync - t_loss_start) * 1000)

                if not hasattr(model, '_timing_stats'):
                    model._timing_stats = {'loss_ms': [], 'bw_cpu_ms': [], 'bw_total_ms': []}
                model._timing_stats['loss_ms'].append(loss_time)
                model._timing_stats['bw_cpu_ms'].append((t_bw_nosync - t_bw_start) * 1000)
                model._timing_stats['bw_total_ms'].append((t_bw_sync - t_bw_start) * 1000)
            if cfg.export_trace:
                prof_ctx.step()

    measured_iters = len(fw_times)

    print("=== Qwen Profile (Genesis) ===")
    print(f"Batch={cfg.batch} SeqLen={cfg.seqlen} Layers={cfg.layers} Dtype={cfg.dtype} Device={cfg.device}")
    print(f"Warmup: {cfg.warmup} iters | Stabilize: {cfg.stabilize} iters | Measured: {measured_iters} iters")

    fw_stats = compute_time_stats(fw_times)
    bw_stats = compute_time_stats(bw_times)
    loss_stats = compute_time_stats(loss_times)

    stage_stats = collect_stage_stats(model.timer, measured_iters)
    fw_mean = fw_stats.get('mean', 0.0)

    highlights: List[str] = []
    denom = max(1, measured_iters)
    if stage_stats and fw_mean:
        top_stage = stage_stats[0]
        share_fw = 100.0 * top_stage.per_iter_ms / fw_mean if fw_mean else 0.0
        highlights.append(f"Forward: {top_stage.name} {top_stage.per_iter_ms:.2f} ms/iter ({share_fw:.1f}% fw)")
    if cfg.deep and DeepStats.linear:
        linear_sorted = sorted(DeepStats.linear.items(), key=lambda kv: kv[1]['ms'], reverse=True)
        top_name, top_ent = linear_sorted[0]
        per_iter = top_ent['ms'] / denom
        share = 100.0 * per_iter / fw_mean if fw_mean else 0.0
        highlights.append(f"Linear: {top_name} {per_iter:.2f} ms/iter ({share:.1f}% fw)")
    if cfg.deep and DeepStats.sdpa_calls:
        sdpa_iter = DeepStats.sdpa_ms / denom
        share = 100.0 * sdpa_iter / fw_mean if fw_mean else 0.0
        highlights.append(f"SDPA: {sdpa_iter:.2f} ms/iter ({share:.1f}% fw)")
    if cfg.profile_backward and BackwardStats.op:
        op_name, ent = sorted(BackwardStats.op.items(), key=lambda kv: kv[1]['ms'], reverse=True)[0]
        per_iter = ent['ms'] / denom
        avg = ent['ms'] / max(1, ent['calls'])
        highlights.append(f"Backward: {op_name} {per_iter:.2f} ms/iter (avg {avg:.2f} ms)")
    if ContiguousStats.calls:
        mb = ContiguousStats.bytes / (1024 * 1024)
        highlights.append(f"Contiguous copies: {ContiguousStats.calls/denom:.1f}/iter ({mb:.1f} MB total)")

    if highlights:
        print("\nBottleneck Highlights:")
        for item in highlights[:5]:
            print(f"  - {item}")

    print_time_stats("Forward Pass", fw_stats)
    if bw_stats:
        print_time_stats("Backward Pass (incl. loss)", bw_stats)
    if loss_stats:
        print_time_stats("Loss Computation", loss_stats)

    print_stage_stats(stage_stats, fw_mean)

    if ContiguousStats.calls:
        mb = ContiguousStats.bytes / (1024 * 1024)
        per_iter_calls = ContiguousStats.calls / denom
        print("\nNon-contiguous -> contiguous copies:")
        print(f"  calls: {ContiguousStats.calls} ({per_iter_calls:.1f} per iter) | bytes: {mb:.1f} MB")
        if ContiguousStats.sources:
            print("\n  Top sources of contiguous copies:")
            sorted_sources = sorted(ContiguousStats.sources.items(), key=lambda x: x[1], reverse=True)
            for source, count in sorted_sources[:10]:
                print(f"    {source}: {count} calls")

    if cfg.profile_backward and hasattr(model, '_timing_stats'):
        stats = model._timing_stats
        if stats['loss_ms']:
            avg_loss = sum(stats['loss_ms']) / len(stats['loss_ms'])
            avg_bw_cpu = sum(stats['bw_cpu_ms']) / len(stats['bw_cpu_ms'])
            avg_bw_total = sum(stats['bw_total_ms']) / len(stats['bw_total_ms'])
            avg_bw_gpu = avg_bw_total - avg_bw_cpu

            print("\n=== CPU vs GPU Time Breakdown (averages) ===")
            print(f"Loss computation:     {avg_loss:.2f} ms")
            print(f"Backward CPU time:    {avg_bw_cpu:.2f} ms (graph construction)")
            print(f"Backward GPU time:    {avg_bw_gpu:.2f} ms (kernel execution)")
            print(f"Backward total:       {avg_bw_total:.2f} ms")
            print(f"CPU overhead ratio:   {100 * avg_bw_cpu / avg_bw_total:.1f}%")

    if cfg.show_memory:
        try:
            print("\n=== CUDA Memory Summary (Genesis) ===")
            print(genesis.cuda.memory_summary())
        except Exception as e:
            print(f"Memory summary not available: {e}")

    if cfg.deep:
        denom = max(1, measured_iters)
        print("\n=== Deep Module Profile ===")
        linear_sorted = sorted(DeepStats.linear.items(), key=lambda kv: kv[1]['ms'], reverse=True)
        if linear_sorted:
            print("Top Linear modules (ms/iter):")
            for name, ent in linear_sorted[:10]:
                total_ms = ent['ms']
                calls = ent['calls']
                per_iter = total_ms / denom
                per_call = total_ms / max(1, calls)
                shapes = ent.get('shapes') or {}
                print(
                    f"  {name:22s} {per_iter:7.2f} ms/iter  avg {per_call:6.2f} ms ({calls:3d}x) shapes={shapes}"
                )
            if len(linear_sorted) > 10:
                rem_total = sum(ent['ms'] for _, ent in linear_sorted[10:])
                print(f"  ... {len(linear_sorted) - 10} more linear modules -> {rem_total / denom:.2f} ms/iter")
        else:
            print("No linear modules captured.")

        if DeepStats.embedding:
            print("\nEmbedding modules (ms/iter):")
            emb_sorted = sorted(DeepStats.embedding.items(), key=lambda kv: kv[1]['ms'], reverse=True)
            for name, ent in emb_sorted[:10]:
                per_iter = ent['ms'] / denom
                per_call = ent['ms'] / max(1, ent['calls'])
                print(f"  {name:22s} {per_iter:7.2f} ms/iter  avg {per_call:6.2f} ms ({ent['calls']}x)")
            if len(DeepStats.embedding) > 10:
                rem_total = sum(ent['ms'] for _, ent in emb_sorted[10:])
                print(f"  ... {len(DeepStats.embedding) - 10} more embeddings -> {rem_total / denom:.2f} ms/iter")

        if DeepStats.sdpa_calls:
            avg = DeepStats.sdpa_ms / max(1, DeepStats.sdpa_calls)
            per_iter = DeepStats.sdpa_ms / denom
            print(f"\nSDPA attention: {per_iter:.2f} ms/iter (avg {avg:.2f} ms across {DeepStats.sdpa_calls} calls)")

        print(f"\nLinear non-fp16 calls: {DeepStats.non_fp16_linear} | non-contiguous inputs: {DeepStats.non_contig_inputs}")

        if cfg.profile_backward:
            print("\n=== Backward Op Profile ===")
            if BackwardStats.op:
                op_sorted = sorted(BackwardStats.op.items(), key=lambda kv: kv[1]['ms'], reverse=True)
                print("Top ops (ms/iter):")
                for name, ent in op_sorted[:20]:
                    per_iter = ent['ms'] / denom
                    per_call = ent['ms'] / max(1, ent['calls'])
                    print(f"  {name:20s} {per_iter:7.2f} ms/iter  avg {per_call:6.2f} ms ({ent['calls']:3d}x)")
            if BackwardStats.gemm:
                print("\nBackward GEMM (by shape):")
                gemm_sorted = sorted(BackwardStats.gemm.items(), key=lambda kv: kv[1]['ms'], reverse=True)
                for k, ent in gemm_sorted[:20]:
                    per_iter = ent['ms'] / denom
                    per_call = ent['ms'] / max(1, ent['calls'])
                    print(f"  {k:40s} {per_iter:7.2f} ms/iter  avg {per_call:6.2f} ms ({ent['calls']:3d}x)")
            print(f"\nBackward non-fp16 matmul calls: {BackwardStats.non_fp16_matmul}")

    # Restore sdpa
    if orig_sdpa is not None:
        import genesis.nn.functional as NF
        NF.scaled_dot_product_attention = orig_sdpa
if __name__ == '__main__':
    args = build_args()
    run(args)
