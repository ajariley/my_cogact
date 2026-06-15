import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# When this file is launched as `python distillation/test.py`, importing through the
# `distillation` package triggers `distillation/__init__.py`, which imports train.py
# and can initialize the default process group before this script runs.
from action_model.action_model import ActionModel
from checkpoint import load_checkpoint
from loaders import load_student, load_teacher


def log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def sync_point(rank: int, label: str) -> None:
    torch.cuda.synchronize()
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    log(rank, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal staged DDP diagnostic for CogACT distillation.")
    parser.add_argument(
        "--stage",
        choices=("linear", "student", "student_ckpt", "teacher_student"),
        default="linear",
        help="How far to go in the loading path.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("/home/huangjiaqi/projects/CogACT-Base/checkpoints/CogACT-Base.pt"),
        help="Teacher checkpoint path used by load_teacher.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional distillation checkpoint path used by load_checkpoint for the student.",
    )
    parser.add_argument("--action-model-type-teacher", default="DiT-B")
    parser.add_argument("--action-model-type-student", default="DiT-S")
    parser.add_argument("--future-action-window-size", type=int, default=15)
    parser.add_argument("--past-action-window-size", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--use-bf16", action="store_true", help="Cast teacher.vlm to bf16 as train.py does.")
    parser.add_argument(
        "--student-ddp-target",
        choices=("net", "full", "x_embedder"),
        default="net",
        help="Which student module to wrap with DDP.",
    )
    parser.add_argument("--skip-create-ddim", action="store_true", help="Construct ActionModel without create_ddim().")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--broadcast-buffers", action="store_true")
    return parser.parse_args()


def build_student(args: argparse.Namespace, token_size: int, device: torch.device):
    if args.skip_create_ddim:
        student = ActionModel(
            token_size=token_size,
            model_type=args.action_model_type_student,
            in_channels=args.action_dim,
            future_action_window_size=args.future_action_window_size,
            past_action_window_size=args.past_action_window_size,
        ).to(device)
        return student
    return load_student(
        token_size=token_size,
        action_model_type=args.action_model_type_student,
        in_channels=args.action_dim,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=args.past_action_window_size,
        device=device,
    )


def main() -> None:
    args = parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    log(rank, f"init ok | local_rank={local_rank} world_size={world_size} stage={args.stage}")

    if args.stage == "linear":
        model = torch.nn.Linear(10, 10).to(device)
        sync_point(rank, "linear model on device")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        sync_point(rank, "linear model wrapped with DDP")
        x = torch.randn(4, 10, device=device)
        y = model(x).sum()
        sync_point(rank, "linear forward ok")
        y.backward()
        sync_point(rank, "linear backward ok")
        dist.destroy_process_group()
        log(rank, "done")
        return

    teacher = None
    if args.stage == "teacher_student":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_bf16):
            teacher = load_teacher(
                args.teacher_checkpoint,
                args.action_model_type_teacher,
                args.future_action_window_size,
            )
        sync_point(rank, "teacher loaded on CPU")
        if args.use_bf16:
            teacher.vlm = teacher.vlm.to(torch.bfloat16)
            sync_point(rank, "teacher VLM cast to bf16")
        teacher = teacher.to(device)
        sync_point(rank, "teacher moved to device")
        token_size = teacher.llm_backbone.llm.lm_head.in_features
    else:
        token_size = 4096

    student = build_student(args, token_size, device)
    sync_point(rank, "student loaded on device")

    ddp_kwargs = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "find_unused_parameters": args.find_unused_parameters,
        "broadcast_buffers": args.broadcast_buffers,
    }

    if args.student_ddp_target == "net":
        student.net = DDP(student.net, **ddp_kwargs)
        sync_point(rank, "student.net wrapped with DDP")
    elif args.student_ddp_target == "full":
        student = DDP(student, **ddp_kwargs)
        sync_point(rank, "student full module wrapped with DDP")
    else:
        student.net.x_embedder = DDP(student.net.x_embedder, **ddp_kwargs)
        sync_point(rank, "student.net.x_embedder wrapped with DDP")

    if args.stage in {"student_ckpt", "teacher_student"} and args.resume_checkpoint is not None:
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        load_checkpoint(args.resume_checkpoint, student, optimizer, map_location=device)
        sync_point(rank, f"student checkpoint loaded from {args.resume_checkpoint}")

    x = torch.randn(4, args.action_dim, device=device)
    if args.student_ddp_target == "full":
        timestep = torch.randint(0, 100, (x.size(0),), device=device)
        z = torch.randn(x.size(0), 1, token_size, device=device)
        out = student.module.net(x, timestep, z).sum()
        sync_point(rank, "student full forward ok")
        out.backward()
        sync_point(rank, "student full backward ok")
    else:
        out = student.net.x_embedder(x).sum()
        sync_point(rank, "student partial forward ok")
        out.backward()
        sync_point(rank, "student partial backward ok")

    dist.destroy_process_group()
    log(rank, "done")


if __name__ == "__main__":
    main()
