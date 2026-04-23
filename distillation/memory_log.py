import os
import subprocess
from typing import List, Optional

# Intentionally no `import torch` at module level so this file can be loaded via
# importlib before torch/tensorflow for baseline snapshots.

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime fallback
    psutil = None


def _process_rank_parts() -> List[str]:
    """Torchrun sets RANK / LOCAL_RANK; include in log so multi-proc logs are distinguishable."""
    r, lr = os.environ.get("RANK"), os.environ.get("LOCAL_RANK")
    if r is not None and lr is not None:
        return [f"rank={r}", f"local_rank={lr}"]
    return []


def _bytes_to_mib_str(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f}MiB"


def _cpu_rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process(os.getpid()).memory_info().rss)

    # Linux fallback when psutil is not installed.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024  # kB -> bytes
    except Exception:
        pass
    return -1


def _cuda_visible_device_specs_for_nvidia_smi() -> Optional[List[str]]:
    """
    若设置了 CUDA_VISIBLE_DEVICES，则只查这些物理下标；
    若未设置，返回 None → nvidia-smi 不指定 -i，查询本机全部 GPU。
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _first_cuda_visible_device_spec() -> str:
    """用于 PyTorch 当前进程视角的说明字段。"""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return "unset(all_physical)"
    specs = [x.strip() for x in raw.split(",") if x.strip()]
    return specs[0] if specs else "0"


def _nvidia_smi_memory_line() -> Optional[str]:
    """
    每张卡一行合并到一条字符串：物理 index、剩余显存 free、总量 total、已用 used（MiB）。
    未设置 CUDA_VISIBLE_DEVICES 时查询机器上全部 GPU。
    """
    specs = _cuda_visible_device_specs_for_nvidia_smi()
    cmd: List[str] = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if specs is not None:
        if not specs:
            return None
        cmd.extend(["-i", ",".join(specs)])
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        chunks: List[str] = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx, free_m, used_m, total_m = parts[0], parts[1], parts[2], parts[3]
                chunks.append(
                    f"gpu_phy{idx}:remaining_MiB={free_m} total_MiB={total_m} used_MiB={used_m}"
                )
        return " | ".join(chunks) if chunks else None
    except Exception:
        pass
    return None


def log_memory_process_start(tag: str = "process_start_before_heavy_imports") -> None:
    """
    Baseline snapshot without importing PyTorch. Call at the very top of the entry script
    (after env vars only) to see CPU RSS and GPU usage before torch/tf/prismatic imports.

    Under torchrun, every rank prints one line (rank/local_rank) and nvidia-smi for each GPU
    (all physical GPUs if CUDA_VISIBLE_DEVICES unset; else only listed indices).
    """
    parts: List[str] = [f"[memory:{tag}]"]
    parts.extend(_process_rank_parts())

    rss_bytes = _cpu_rss_bytes()
    if rss_bytes >= 0:
        parts.append(f"cpu_rss={_bytes_to_mib_str(rss_bytes)}")
    else:
        parts.append("cpu_rss=unavailable")

    smi = _nvidia_smi_memory_line()
    if smi:
        parts.append(smi)
    else:
        parts.append("nvidia_smi=unavailable")

    print(" | ".join(parts), flush=True)


def _collect_tf_gpu_memory() -> List[str]:
    lines: List[str] = []
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return ["tf_gpu=none"]
        for i, _ in enumerate(gpus):
            try:
                info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                current = int(info.get("current", -1))
                peak = int(info.get("peak", -1))
                lines.append(
                    f"tf_gpu{i}_current={_bytes_to_mib_str(current)} tf_gpu{i}_peak={_bytes_to_mib_str(peak)}"
                )
            except Exception as e:
                lines.append(f"tf_gpu{i}_memory_info_unavailable={type(e).__name__}")
    except Exception as e:
        lines.append(f"tf_unavailable={type(e).__name__}")
    return lines


def log_memory(tag: str, log_tf: bool = True, empty_cache: bool = False) -> None:
    """
    Log process CPU RSS and GPU memory metrics for quick OOM debugging.
    Imports torch lazily inside this function.
    Each rank prints PyTorch stats for current device, per-logical-GPU mem_get_info, and nvidia-smi per physical GPU.
    """
    import torch

    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    parts: List[str] = [f"[memory:{tag}]"]
    parts.extend(_process_rank_parts())

    rss_bytes = _cpu_rss_bytes()
    if rss_bytes >= 0:
        parts.append(f"cpu_rss={_bytes_to_mib_str(rss_bytes)}")
    else:
        parts.append("cpu_rss=unavailable")

    if torch.cuda.is_available():
        parts.append(f"cuda_visible_hint={_first_cuda_visible_device_spec()}")
        dev = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        allocated = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        parts.extend(
            [
                f"cuda_current_logical={dev}",
                f"cuda_free={_bytes_to_mib_str(free_bytes)}",
                f"cuda_total={_bytes_to_mib_str(total_bytes)}",
                f"torch_allocated={_bytes_to_mib_str(allocated)}",
                f"torch_reserved={_bytes_to_mib_str(reserved)}",
            ]
        )
        # 本进程可见的每一张逻辑卡：剩余 / 总量（与 nvidia-smi 在可见设备限制下一致）
        for d in range(torch.cuda.device_count()):
            f_b, t_b = torch.cuda.mem_get_info(d)
            parts.append(
                f"torch_cuda{d}:remaining={_bytes_to_mib_str(f_b)} total={_bytes_to_mib_str(t_b)}"
            )
    else:
        parts.append("cuda=unavailable")

    smi = _nvidia_smi_memory_line()
    if smi:
        parts.append(smi)
    else:
        parts.append("nvidia_smi=unavailable")

    if log_tf:
        parts.extend(_collect_tf_gpu_memory())

    print(" | ".join(parts), flush=True)
