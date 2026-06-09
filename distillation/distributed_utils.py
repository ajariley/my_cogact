from dataclasses import dataclass
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool


def init_distributed() -> DistInfo:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if is_distributed and dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl")

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
    )


def get_dist_info() -> DistInfo:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = world_size > 1
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        is_distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistInfo(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
    )


def is_main_process() -> bool:
    return get_dist_info().rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_ddp_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module
